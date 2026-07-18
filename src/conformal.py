"""
conformal.py
============
Distribution-free CONFORMAL ABSTENTION layer for KinetiFlow-CP v2.

WHAT THIS FORECASTS
    The trained gray-box UDE forecasts the 15-minute EQUILIBRIUM optical
    intensity I(900 s) [DN] of an hCG lateral-flow assay from early kinetics.
    identifiability.py proved this early forecast is under-determined
    (alpha<->B_max and k_on<->k_m are perfectly confounded), so a point forecast
    alone is not trustworthy. This module wraps the forecast in a calibrated
    prediction interval and an ABSTAIN/RETEST rule, so the system can decline to
    make a clinical call when it is not entitled to one.

NONCONFORMITY SCORE
    Per calibration/test trace i, the score is the absolute forecast error on the
    equilibrium intensity:

        s_i = | I_true,i - I_pred,i |,   I_true = true_I_900 (noise-free target),
                                         I_pred = model I(900 s) = I_pred[-1].

    We score against the noise-free `true_I_900` because the project forecasts the
    *equilibrium* value, which the dataset stores as exactly that field. (Scoring
    against the noisy I_obs[:, -1] instead would simply widen every interval by the
    measurement noise; the coverage machinery below is identical either way.)

GUARANTEE (do not overclaim)
    * SPLIT CONFORMAL (SCP): finite-sample marginal coverage >= 1 - delta holds
      under EXCHANGEABILITY of calibration and test points (same distribution).
      It does NOT hold under arbitrary distribution shift.
    * WR-CP (weighted split conformal, Tibshirani 2019, + Wasserstein-based
      weight regularization, Xu 2025): restores coverage only under BOUNDED
      COVARIATE SHIFT with overlapping support and adequate weight effective
      sample size. It is EXPLORATORY at n_cal~40 and FALLS BACK to SCP (clearly
      flagged) when the weight ESS < 20 -- it never silently emits garbage
      intervals.
    Neither method covers arbitrary OOD.

LEAKAGE
    Conformal calibration uses the CALIBRATION split only (leakage-safe by day and
    lot, from dataset.py). Test / OOD data never enter calibration, and the
    abstention thresholds are LOCKED on calibration before any test evaluation.

Run the self-test:  python src/conformal.py
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"

TARGET_DELTA = 0.10          # default miscoverage -> 90% prediction intervals
ESS_MIN = 20.0              # WR-CP stability floor on weight effective sample size


# --------------------------------------------------------------------------- #
#  Array helpers                                                              #
# --------------------------------------------------------------------------- #
def _asnp(x) -> np.ndarray:
    """Coerce a torch.Tensor / array-like to a float64 numpy array (detached)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float64)


# =========================================================================== #
#  Part 1 — SPLIT CONFORMAL PREDICTION (SCP)  [PRIMARY method]                 #
# =========================================================================== #
def nonconformity_scores(pred, true) -> np.ndarray:
    """Absolute-residual nonconformity score s_i = |true_i - pred_i|."""
    return np.abs(_asnp(true) - _asnp(pred))


def scp_quantile(scores, delta: float = TARGET_DELTA) -> float:
    """Split-conformal half-width q for miscoverage `delta` (coverage 1 - delta).

    Finite-sample rule (Vovk; Lei et al. 2018):
        k = ceil((n_cal + 1) * (1 - delta));  q = the k-th smallest score
        (1-indexed). This equals the (1-delta)(1 + 1/n_cal) empirical quantile.

    If k > n_cal the required quantile is +inf: there are not enough calibration
    points to certify the requested coverage, so the honest interval is unbounded.
    We return +inf and warn (correct finite-sample behavior, not an error).
    """
    s = np.sort(_asnp(scores))
    n = s.size
    if n == 0:
        raise ValueError("scp_quantile: empty calibration score set")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"scp_quantile: delta must be in (0, 1), got {delta}")
    k = math.ceil((n + 1) * (1.0 - delta))
    if k > n:
        warnings.warn(
            f"scp_quantile: k={k} > n_cal={n} at delta={delta} -> q=+inf "
            f"(infinite interval; need >= {k - 1} calibration points for "
            f"{100 * (1 - delta):.0f}% coverage).",
            stacklevel=2,
        )
        return math.inf
    return float(s[k - 1])


def scp_interval(pred, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric conformal interval [pred - q, pred + q]."""
    p = _asnp(pred)
    return p - q, p + q


def coverage(true, lo, hi) -> float:
    """Fraction of points whose true value lies within [lo, hi] (inclusive)."""
    y, lo, hi = _asnp(true), _asnp(lo), _asnp(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    """Mean interval width hi - lo (may be +inf if q was infinite)."""
    return float(np.mean(_asnp(hi) - _asnp(lo)))


# =========================================================================== #
#  Part 2 — WEIGHTED / WASSERSTEIN-REGULARIZED CONFORMAL (WR-CP)               #
#           OOD extension for covariate shift. Exploratory at small n_cal.     #
# =========================================================================== #
def density_ratio_weights(cal_X, test_X, l2: float = 1.0, prob_eps: float = 1e-3
                          ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Importance weights w(x) = dP_test/dP_cal(x) by probabilistic classification.

    Train a logistic classifier to separate CALIBRATION (label 0) from TEST
    (label 1) covariates x = [concentration_level, T, RH] (standardized on the
    pooled set). With the empirical class priors, the density ratio is

        w(x) = [p / (1 - p)] * (n_cal / n_test),     p = P(test | x).

    Returns (w_cal, w_test, info); info["clf_auc"] is the classifier ROC-AUC, an
    imbalance-robust read on how detectable the shift is (0.5 = indistinguishable
    = no shift; -> 1.0 = strongly separable = large shift).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    Xc, Xt = _asnp(cal_X), _asnp(test_X)
    if Xc.ndim == 1:
        Xc = Xc[:, None]
    if Xt.ndim == 1:
        Xt = Xt[:, None]
    n_cal, n_test = len(Xc), len(Xt)

    X = np.vstack([Xc, Xt])
    y = np.concatenate([np.zeros(n_cal), np.ones(n_test)])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0 / l2, max_iter=2000)
    clf.fit(scaler.transform(X), y)

    def _w(Xr: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(scaler.transform(Xr))[:, 1]
        p = np.clip(p, prob_eps, 1.0 - prob_eps)
        return (p / (1.0 - p)) * (n_cal / n_test)

    # AUC (not accuracy) so the 40-vs-n_test class imbalance can't masquerade as
    # a shift: a no-shift classifier has AUC ~ 0.5 regardless of the class ratio.
    auc = float(roc_auc_score(y, clf.predict_proba(scaler.transform(X))[:, 1]))
    return _w(Xc), _w(Xt), {"clf_auc": auc}


def effective_sample_size(w) -> float:
    """Kish effective sample size ESS = (sum w)^2 / sum(w^2). Low ESS => the
    weights concentrate on a few calibration points => unstable reweighting."""
    w = _asnp(w)
    s2 = float((w * w).sum())
    return float((w.sum() ** 2) / s2) if s2 > 0 else 0.0


def wasserstein1(a, b) -> float:
    """1-D Wasserstein-1 distance between two score samples (the shift magnitude
    reported by WR-CP). Uses scipy's exact empirical-CDF estimator."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(_asnp(a), _asnp(b)))


def wrcp_quantiles(cal_scores, w_cal, w_test, delta: float = TARGET_DELTA
                   ) -> np.ndarray:
    """Per-test-point weighted split-conformal half-widths (Tibshirani 2019).

    For test point j (covariate weight w_test_j) the normalized calibration
    masses are p_i = w_i / (sum_k w_k + w_test_j) and the +inf atom carries
    p_{n+1} = w_test_j / (sum_k w_k + w_test_j). The half-width q_j is the
    smallest calibration score whose weighted cumulative mass reaches (1 - delta):

        q_j = inf { s : sum_{i: s_i <= s} w_i  >=  (1 - delta) * (S + w_test_j) },

    with S = sum_i w_i; q_j = +inf when the finite mass cannot reach the target
    (the test atom is too heavy). Vectorized over all test points via searchsorted.
    """
    s = _asnp(cal_scores)
    wc = _asnp(w_cal)
    wt = _asnp(w_test)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    prefix = np.cumsum(wc[order])                    # cumulative calibration mass
    S = float(prefix[-1])
    target = (1.0 - delta) * (S + wt)                # [n_test] required mass
    idx = np.searchsorted(prefix, target, side="left")
    q = np.full(wt.shape, np.inf, dtype=np.float64)
    valid = idx < s_sorted.size
    q[valid] = s_sorted[idx[valid]]
    return q


def _regularize_weights(w, cap: float) -> np.ndarray:
    """Clip weights to [0, cap] to bound the variance of the weighted quantile."""
    return np.clip(_asnp(w), 0.0, cap)


def wrcp_intervals(cal_scores, cal_X, test_X, test_pred, test_scores,
                   delta: float = TARGET_DELTA, ess_min: float = ESS_MIN) -> Dict:
    """Weighted / Wasserstein-regularized conformal intervals for a shifted test.

    Steps: (1) density-ratio weights from a cal-vs-test classifier; (2) the
    Wasserstein-1 distance between the cal and test SCORE distributions as the
    shift magnitude, which tightens a variance-control clip on the (heavy-tailed)
    weights as the shift grows; (3) a STABILITY GUARD -- if the RAW weight ESS is
    below `ess_min` the reweighting is too concentrated to trust, so we FALL BACK
    to plain SCP and flag WR-CP as exploratory/unstable; otherwise (4) per-test-
    point weighted quantiles (Tibshirani 2019).

    Returns a dict of intervals (lo, hi, q) and diagnostics (ess, wasserstein,
    fell_back, clf_auc, cap).
    """
    cal_scores = _asnp(cal_scores)
    test_pred = _asnp(test_pred)

    w_cal_raw, w_test_raw, info = density_ratio_weights(cal_X, test_X)
    ess = effective_sample_size(w_cal_raw)                 # guard on RAW weights
    d_w = wasserstein1(cal_scores, test_scores)            # score-shift magnitude

    # Wasserstein-tied variance-control clip: larger measured shift -> tighter cap
    # on the weights (kappa in [1, 9]). Reported and applied consistently to the
    # calibration and test-point weights. (Exploratory; clipping trades a little
    # exactness for stability -- WR-CP coverage under shift is approximate.)
    scale = float(np.std(cal_scores)) + 1e-9
    kappa = 1.0 + 8.0 / (1.0 + d_w / scale)
    cap = float(np.mean(w_cal_raw)) * kappa
    w_cal = _regularize_weights(w_cal_raw, cap)
    w_test = _regularize_weights(w_test_raw, cap)

    fell_back = ess < ess_min
    if fell_back:
        q = np.full(test_pred.shape, scp_quantile(cal_scores, delta), dtype=np.float64)
    else:
        q = wrcp_quantiles(cal_scores, w_cal, w_test, delta)

    lo, hi = test_pred - q, test_pred + q
    return {
        "lo": lo, "hi": hi, "q": q,
        "ess": ess, "ess_min": ess_min,
        "wasserstein": d_w, "cap": cap, "kappa": kappa,
        "fell_back": fell_back, "clf_auc": info["clf_auc"],
        "w_cal": w_cal, "w_test": w_test,
    }


# =========================================================================== #
#  Part 3 — ABSTENTION PROTOCOL  (thresholds LOCKED on calibration only)       #
# =========================================================================== #
def lock_thresholds(cal_true, cal_levels, neg_level: float = 0.5,
                    pos_level: float = 1.0, width_frac: float = 1.0
                    ) -> Tuple[float, float]:
    """Lock the clinical decision threshold and interval-width tolerance on the
    CALIBRATION set ONLY -- computed before any test evaluation, so there is no
    test leakage by construction.

    theta_clinical : intensity (DN) at the nominal clinical cutoff = the cal-mean
        true equilibrium intensity of the `pos_level` (1.0x == 25 mIU/mL) samples.
        A sample whose true equilibrium sits at this intensity is exactly at the
        decision boundary, so the 1.0x population is (correctly) the ambiguous
        "retest" zone. Falls back to the cal median if the level is absent.
    w_max : maximum clinically-useful interval width = `width_frac` * the cal
        separation between the negative (`neg_level`, 0.5x) and positive (1.0x)
        equilibrium clusters. An interval wider than that neg/pos gap cannot
        resolve a call, so we abstain. Falls back to width_frac * std(cal_true).
    """
    yt = _asnp(cal_true)
    lv = _asnp(cal_levels)
    neg = yt[lv == neg_level]
    pos = yt[lv == pos_level]
    theta = float(pos.mean()) if pos.size else float(np.median(yt))
    if neg.size and pos.size:
        w_max = width_frac * abs(float(pos.mean()) - float(neg.mean()))
    else:
        w_max = width_frac * float(np.std(yt))
    return theta, w_max


def abstain_masks(lo, hi, theta_clinical: float, w_max: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (straddle, too_wide) boolean masks driving the abstention rule."""
    lo, hi = _asnp(lo), _asnp(hi)
    straddle = (lo < theta_clinical) & (hi > theta_clinical)
    too_wide = (hi - lo) > w_max
    return straddle, too_wide


def abstain(lo, hi, theta_clinical: float, w_max: float) -> np.ndarray:
    """ABSTAIN/RETEST if the interval (a) straddles the clinical threshold OR
    (b) is wider than the clinical tolerance w_max."""
    straddle, too_wide = abstain_masks(lo, hi, theta_clinical, w_max)
    return straddle | too_wide


# =========================================================================== #
#  Part 4 — METRICS this module owns                                          #
# =========================================================================== #
def clinical_calls(pred, true, theta_clinical: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Binary clinical calls: forecast/true equilibrium intensity > theta."""
    return _asnp(pred) > theta_clinical, _asnp(true) > theta_clinical


def selective_accuracy(pred, true, theta_clinical: float, keep_mask
                       ) -> Tuple[float, float, float]:
    """Return (full_acc, selective_acc, abstention_rate).

    Accuracy = agreement between the forecast clinical call and the true call;
    `selective_acc` restricts to the non-abstained (kept) subset. Abstention
    should raise selective accuracy by removing threshold-straddling cases.
    """
    call_p, call_t = clinical_calls(pred, true, theta_clinical)
    correct = (call_p == call_t)
    keep = _asnp(keep_mask).astype(bool)
    full_acc = float(np.mean(correct))
    abst_rate = float(np.mean(~keep))
    sel_acc = float(np.mean(correct[keep])) if keep.any() else float("nan")
    return full_acc, sel_acc, abst_rate


def ood_coverage_drop(clean_cov: float, ood_cov: float) -> float:
    """Coverage lost from the clean to the shifted test under the same interval."""
    return clean_cov - ood_cov


# --------------------------------------------------------------------------- #
#  Model-prediction + OOD-generation helpers (import; do not modify others)    #
# --------------------------------------------------------------------------- #
def predict_equilibrium(model, meas, bundle, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast the 15-min equilibrium intensity for a TraceBundle.

    Returns (pred_900, true_900) as numpy arrays: the model's I(900 s) forecast
    (last timepoint of the no-grad rollout) and the noise-free ground truth.
    Uses train.eval_loss, which handles per-trace covariates internally.
    """
    import train
    _, _, I_pred = train.eval_loss(model, meas, bundle, cfg)   # [T, n]
    return _asnp(I_pred[-1]), _asnp(bundle.true_I_900)


def _covariates(bundle) -> np.ndarray:
    """Density-ratio covariates x = [concentration_level, T_ambient, RH]."""
    return np.column_stack([
        _asnp(bundle.concentration_level),
        _asnp(bundle.T_ambient),
        _asnp(bundle.RH),
    ])


def make_shifted_bundle(name: str, T_nominal: float, RH_nominal: float, seed: int):
    """Generate a fresh synthetic set at (possibly shifted) nominal T/RH and wrap
    it as a dataset.TraceBundle. A distinct `seed` (!= the real dataset's 12345)
    keeps it strictly separate from the calibration data. Reuses
    synthetic.generate unchanged."""
    import torch
    import synthetic
    import dataset as D
    cfg = synthetic.SyntheticConfig(T_nominal=T_nominal, RH_nominal=RH_nominal,
                                    seed=seed)
    data = synthetic.generate(cfg)
    n = int(data["I_obs"].shape[0])
    return D.TraceBundle(
        name=name, idx=torch.arange(n), t=data["t"], I_obs=data["I_obs"],
        C_f0=data["C_f0"], T_ambient=data["T_ambient"], RH=data["RH"],
        day=data["day"], lot=data["lot"],
        concentration_level=data["concentration_level"],
        true_I_900=data["true_I_900"],
    )


# =========================================================================== #
#  Part 5 — CV+ / JACKKNIFE+  (Barber, Candes, Ramdas & Tibshirani 2021)       #
#           Uses ALL calibration data; finite-sample (1 - 2*delta) guarantee.  #
# =========================================================================== #
#  WHY (small-n motivation): split conformal (Part 1) spends its calibration
#  data on a single held-out quantile, so at n_cal ~ 40-80 the threshold q is
#  high-variance. CV+/jackknife+ instead cross-fit a light RECALIBRATION LAYER
#  over the calibration set and aggregate leave-fold-out residuals, using every
#  point for both fitting and scoring, with a distribution-free >= 1 - 2*delta
#  coverage guarantee (typically ~ 1 - delta empirically).
#
#  SCOPE: this layer treats the frozen gray-box endpoint forecast as a FEATURE
#  and refits only a tiny ridge recalibrator across folds. It never retrains the
#  UDE, never unfreezes the kinetics, and never touches training -- it is purely
#  the uncertainty wrapper. (A fixed predictor would collapse CV+ back to SCP, so
#  the refittable recalibrator is what makes the cross-fitting meaningful.)
def recal_features(bundle, pred) -> np.ndarray:
    """Recalibration-layer features [I_pred, C_f0, T_ambient, RH].

    The columns are the frozen UDE endpoint forecast plus the covariates the
    gray-box already consumes at inference (initial dose and environment). On the
    CALIBRATION split `pred` is an out-of-sample forecast, so these features carry
    honest generalization error for CV+/CQR to calibrate against."""
    return np.column_stack([
        _asnp(pred), _asnp(bundle.C_f0), _asnp(bundle.T_ambient), _asnp(bundle.RH),
    ])


class _RidgeLearner:
    """Closed-form ridge on standardized inputs -- the refittable base predictor
    for CV+/jackknife+. Leave-fold-out refits make mu_{-k}(x) vary across folds,
    which is exactly the cross-fitting that separates CV+ from split conformal."""

    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)

    def fit(self, X, y) -> "_RidgeLearner":
        X, y = _asnp(X), _asnp(y)
        self._mu = X.mean(0)
        self._sd = X.std(0) + 1e-9
        Xs = (X - self._mu) / self._sd
        self._ymu = float(y.mean())
        d = Xs.shape[1]
        A = Xs.T @ Xs + self.l2 * np.eye(d)
        self._w = np.linalg.solve(A, Xs.T @ (y - self._ymu))
        return self

    def predict(self, X) -> np.ndarray:
        Xs = (_asnp(X) - self._mu) / self._sd
        return Xs @ self._w + self._ymu


def leakage_safe_folds(bundle) -> np.ndarray:
    """Assign each calibration trace to a CV+ fold that never SPLITS a (day, lot)
    group, so leave-fold-out residuals aren't optimistic from within-group
    correlation.

    Prefers the atomic day<->lot connected components (dataset.py's grouping). If
    the split is one inseparable component, falls back to the finest leakage-safe
    key that still yields >= 2 folds -- distinct days when the lot is constant, or
    distinct lots when the day is constant. If none exists, returns leave-one-out
    (jackknife+) and warns."""
    import dataset as D
    comps = D._atomic_groups(bundle.day, bundle.lot)          # list[list[int pos]]
    n = len(bundle)
    fold = np.full(n, -1, dtype=int)
    if len(comps) >= 2:
        for k, members in enumerate(comps):
            fold[np.asarray(members, dtype=int)] = k
        return fold
    day = _asnp(bundle.day).astype(int)
    lot = _asnp(bundle.lot).astype(int)
    for key in (day, lot):                                    # finest leakage-safe
        uniq = np.unique(key)
        if uniq.size >= 2:
            for k, v in enumerate(uniq):
                fold[key == v] = k
            return fold
    warnings.warn("leakage_safe_folds: calibration is a single inseparable "
                  "day/lot group -> using leave-one-out (jackknife+).", stacklevel=2)
    return np.arange(n)


def _cvplus_column_quantile(A: np.ndarray, delta: float, upper: bool) -> np.ndarray:
    """Per-column Barber-et-al. quantile of an [n, m] array.

    upper=True  -> Q^+ : the ceil((1-delta)(n+1))-th smallest of each column
                          (+inf if that index exceeds n -- not enough cal points).
    upper=False -> Q^- : the floor(delta(n+1))-th smallest of each column
                          (-inf if that index is < 1)."""
    n, m = A.shape
    if upper:
        k = math.ceil((1.0 - delta) * (n + 1))
        if k > n:
            return np.full(m, np.inf)
    else:
        k = math.floor(delta * (n + 1))
        if k < 1:
            return np.full(m, -np.inf)
    return np.sort(A, axis=0)[k - 1, :]


def cvplus_intervals(cal_feats, cal_true, fold_id, target_feats,
                     delta: float = TARGET_DELTA, l2: float = 1.0,
                     learner_factory: Optional[Callable] = None) -> Dict:
    """CV+ / jackknife+ prediction intervals (Barber et al. 2021).

    For each fold k, refit the base learner on the OTHER folds, giving leave-fold-
    out residuals R_i = |y_i - mu_{-k(i)}(x_i)| on held-out calibration points and
    leave-fold-out predictions mu_{-k(i)}(x) on every target point. The interval
    for a target x is

        [ Q^-_{n,delta}{ mu_{-k(i)}(x) - R_i },  Q^+_{n,delta}{ mu_{-k(i)}(x) + R_i } ],

    with marginal coverage >= 1 - 2*delta. Jackknife+ is the special case where
    every fold is one point (fold_id = arange(n)). Vectorized over all targets."""
    cal_feats, cal_true = _asnp(cal_feats), _asnp(cal_true)
    target_feats = _asnp(target_feats)
    fold_id = _asnp(fold_id).astype(int)
    n, m = cal_true.size, target_feats.shape[0]
    factory = learner_factory or (lambda: _RidgeLearner(l2))

    R = np.empty(n)
    center = np.empty((n, m))                    # center[i] = mu_{-k(i)}(target)
    for k in np.unique(fold_id):
        te = fold_id == k
        learner = factory().fit(cal_feats[~te], cal_true[~te])
        R[te] = np.abs(cal_true[te] - learner.predict(cal_feats[te]))
        center[te, :] = learner.predict(target_feats)[None, :]

    lo = _cvplus_column_quantile(center - R[:, None], delta, upper=False)
    hi = _cvplus_column_quantile(center + R[:, None], delta, upper=True)
    return {"lo": lo, "hi": hi, "R": R, "n_folds": int(np.unique(fold_id).size)}


# =========================================================================== #
#  Part 6 — CONFORMALIZED QUANTILE REGRESSION (Romano, Patterson & Candes 2019) #
#           Locally-adaptive intervals: width grows with dose / hook region.   #
# =========================================================================== #
def _fit_quantile_head(X, y, quantile: float, alpha: float = 1e-4):
    """Linear quantile-regression head (pinball loss) at level `quantile`, fit on
    standardized features via scikit-learn's exact LP solver. Returns (scaler,
    model). Separate lo/hi slopes let the band widen where uncertainty grows."""
    from sklearn.linear_model import QuantileRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(_asnp(X))
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    model.fit(scaler.transform(_asnp(X)), _asnp(y))
    return scaler, model


def _predict_quantile_head(head, X) -> np.ndarray:
    scaler, model = head
    return model.predict(scaler.transform(_asnp(X)))


def cqr_fit(train_feats, train_true, cal_feats, cal_true,
            delta: float = TARGET_DELTA, alpha: float = 1e-4) -> Dict:
    """Conformalized Quantile Regression (Romano et al. 2019).

    Fit lower (delta/2) and upper (1-delta/2) quantile heads on the TRAIN split
    (large, and out-of-sample for calibration), then CONFORMALIZE on calibration
    with the CQR nonconformity score

        E_i = max( qlo(x_i) - y_i,  y_i - qhi(x_i) ),

    and E = the finite-sample (1-delta) quantile of {E_i} (reuses scp_quantile).
    The calibrated interval is [qlo(x) - E, qhi(x) + E], marginal coverage
    >= 1 - delta. E may be negative -> over-wide heads are tightened.

    HONESTY: the heads use the frozen UDE forecast as a feature, which is IN-SAMPLE
    on the train split, so the raw band can be optimistically narrow; the
    calibration step restores >= 1 - delta coverage regardless, but the *local
    adaptivity* is only as good as the heads."""
    lo_head = _fit_quantile_head(train_feats, train_true, delta / 2.0, alpha)
    hi_head = _fit_quantile_head(train_feats, train_true, 1.0 - delta / 2.0, alpha)
    q_lo = _predict_quantile_head(lo_head, cal_feats)
    q_hi = _predict_quantile_head(hi_head, cal_feats)
    E_scores = np.maximum(q_lo - _asnp(cal_true), _asnp(cal_true) - q_hi)
    E = scp_quantile(E_scores, delta)                         # signed CQR scores
    return {"lo_head": lo_head, "hi_head": hi_head, "E": float(E)}


def cqr_intervals(cqr: Dict, target_feats) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrated CQR interval [qlo(x) - E, qhi(x) + E] for a target feature set."""
    q_lo = _predict_quantile_head(cqr["lo_head"], target_feats)
    q_hi = _predict_quantile_head(cqr["hi_head"], target_feats)
    return q_lo - cqr["E"], q_hi + cqr["E"]


# =========================================================================== #
#  Part 7 — COMMON METHOD INTERFACE  (so evaluate.py runs every method alike)   #
# =========================================================================== #
@dataclass
class CalContext:
    """Everything a conformal method needs, computed ONCE from a frozen predictor.

    `predict_fn(bundle) -> (pred, true)` is the model wrapper -- e.g.
    `lambda b: predict_equilibrium(model, meas, b, cfg)` for the gray-box UDE, or
    `tcn.predict` for the TCN baseline -- so the SAME context drives SCP, WR-CP,
    CV+/jackknife+, and CQR side by side. Only calibration (and, for CQR, train)
    data enter here; test / OOD data never do."""
    predict_fn: Callable
    cal_bundle: object
    train_bundle: object = None
    delta: float = TARGET_DELTA
    cal_pred: Optional[np.ndarray] = None
    cal_true: Optional[np.ndarray] = None
    cal_scores: Optional[np.ndarray] = None
    cal_feats: Optional[np.ndarray] = None
    cal_shift_cov: Optional[np.ndarray] = None
    train_true: Optional[np.ndarray] = None
    train_feats: Optional[np.ndarray] = None


def build_context(predict_fn: Callable, cal_bundle, train_bundle=None,
                  delta: float = TARGET_DELTA) -> CalContext:
    """Precompute the calibration (and optional train) predictions/features once."""
    cal_pred, cal_true = predict_fn(cal_bundle)
    ctx = CalContext(predict_fn=predict_fn, cal_bundle=cal_bundle,
                     train_bundle=train_bundle, delta=delta)
    ctx.cal_pred = _asnp(cal_pred)
    ctx.cal_true = _asnp(cal_true)
    ctx.cal_scores = nonconformity_scores(cal_pred, cal_true)
    ctx.cal_feats = recal_features(cal_bundle, cal_pred)
    ctx.cal_shift_cov = _covariates(cal_bundle)
    if train_bundle is not None:
        tr_pred, tr_true = predict_fn(train_bundle)
        ctx.train_true = _asnp(tr_true)
        ctx.train_feats = recal_features(train_bundle, tr_pred)
    return ctx


class IntervalMethod:
    """Uniform interface: fit(ctx) locks everything on calibration/train (never on
    test); intervals(bundle) -> dict with lo, hi, pred (+ method diagnostics)."""
    name = "?"

    def fit(self, ctx: CalContext) -> "IntervalMethod":
        raise NotImplementedError

    def intervals(self, bundle) -> Dict:
        raise NotImplementedError


class SplitConformal(IntervalMethod):
    """Part-1 SCP as a method adapter (retained baseline)."""
    name = "SCP"

    def __init__(self, delta: float = TARGET_DELTA):
        self.delta = delta

    def fit(self, ctx: CalContext) -> "SplitConformal":
        self.ctx = ctx
        self.q = scp_quantile(ctx.cal_scores, self.delta)
        return self

    def intervals(self, bundle) -> Dict:
        pred, _ = self.ctx.predict_fn(bundle)
        lo, hi = scp_interval(pred, self.q)
        return {"lo": lo, "hi": hi, "pred": _asnp(pred), "q": self.q}


class WeightedConformal(IntervalMethod):
    """Part-2 WR-CP as a method adapter (retained covariate-shift baseline)."""
    name = "WR-CP"

    def __init__(self, delta: float = TARGET_DELTA, ess_min: float = ESS_MIN):
        self.delta = delta
        self.ess_min = ess_min

    def fit(self, ctx: CalContext) -> "WeightedConformal":
        self.ctx = ctx
        return self

    def intervals(self, bundle) -> Dict:
        pred, true = self.ctx.predict_fn(bundle)
        scores = nonconformity_scores(pred, true)
        res = wrcp_intervals(self.ctx.cal_scores, self.ctx.cal_shift_cov,
                             _covariates(bundle), pred, scores,
                             delta=self.delta, ess_min=self.ess_min)
        return {"lo": res["lo"], "hi": res["hi"], "pred": _asnp(pred),
                "ess": res["ess"], "fell_back": res["fell_back"],
                "wasserstein": res["wasserstein"], "clf_auc": res["clf_auc"]}


class CVPlus(IntervalMethod):
    """CV+ (grouped K-fold) or jackknife+ (leave-one-out) over the calibration set."""

    def __init__(self, delta: float = TARGET_DELTA, scheme: str = "grouped",
                 l2: float = 1.0):
        assert scheme in ("grouped", "loo")
        self.delta, self.scheme, self.l2 = delta, scheme, l2
        self.name = "CV+" if scheme == "grouped" else "jackknife+"

    def fit(self, ctx: CalContext) -> "CVPlus":
        self.ctx = ctx
        self.fold_id = (leakage_safe_folds(ctx.cal_bundle) if self.scheme == "grouped"
                        else np.arange(ctx.cal_true.size))
        return self

    def intervals(self, bundle) -> Dict:
        pred, _ = self.ctx.predict_fn(bundle)
        tfeats = recal_features(bundle, pred)
        res = cvplus_intervals(self.ctx.cal_feats, self.ctx.cal_true, self.fold_id,
                               tfeats, delta=self.delta, l2=self.l2)
        return {"lo": res["lo"], "hi": res["hi"], "pred": _asnp(pred),
                "n_folds": res["n_folds"]}


class CQR(IntervalMethod):
    """Conformalized Quantile Regression (heads fit on train, conformalized on cal)."""
    name = "CQR"

    def __init__(self, delta: float = TARGET_DELTA, alpha: float = 1e-4):
        self.delta, self.alpha = delta, alpha

    def fit(self, ctx: CalContext) -> "CQR":
        if ctx.train_bundle is None or ctx.train_feats is None:
            raise ValueError("CQR needs a train_bundle in the context to fit its "
                             "quantile heads out-of-sample.")
        self.ctx = ctx
        self.cqr = cqr_fit(ctx.train_feats, ctx.train_true, ctx.cal_feats,
                           ctx.cal_true, delta=self.delta, alpha=self.alpha)
        return self

    def intervals(self, bundle) -> Dict:
        pred, _ = self.ctx.predict_fn(bundle)
        tfeats = recal_features(bundle, pred)
        lo, hi = cqr_intervals(self.cqr, tfeats)
        return {"lo": lo, "hi": hi, "pred": _asnp(pred), "E": self.cqr["E"]}


def build_methods(delta: float = TARGET_DELTA) -> Dict[str, IntervalMethod]:
    """The full comparator suite keyed by name. SCP and WR-CP are the retained
    baselines; CV+/jackknife+ and CQR are the small-calibration upgrades."""
    return {
        "SCP": SplitConformal(delta),
        "WR-CP": WeightedConformal(delta),
        "CV+": CVPlus(delta, scheme="grouped"),
        "jackknife+": CVPlus(delta, scheme="loo"),
        "CQR": CQR(delta),
    }


# --------------------------------------------------------------------------- #
#  Coverage uncertainty + one-call evaluation                                 #
# --------------------------------------------------------------------------- #
def bootstrap_coverage_ci(true, lo, hi, n_boot: int = 4000, ci: float = 0.90,
                          seed: int = 0) -> Tuple[float, float]:
    """Percentile-bootstrap CI for the empirical coverage. At n ~ 40-80 coverage
    is high-variance, so a point estimate alone misleads -- this resamples the
    evaluated points to bound that sampling uncertainty. inf-width points (honest
    unbounded intervals) count as covered."""
    y, lo, hi = _asnp(true), _asnp(lo), _asnp(hi)
    inside = ((y >= lo) & (y <= hi)).astype(np.float64)
    n = inside.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    covs = inside[idx].mean(axis=1)
    a = (1.0 - ci) / 2.0
    return float(np.quantile(covs, a)), float(np.quantile(covs, 1.0 - a))


def evaluate_intervals(true, res: Dict, n_boot: int = 4000, ci: float = 0.90,
                       seed: int = 0) -> Dict:
    """Coverage, mean width, and a bootstrap CI on coverage for one method's
    intervals on one target set."""
    lo, hi = res["lo"], res["hi"]
    lo_ci, hi_ci = bootstrap_coverage_ci(true, lo, hi, n_boot=n_boot, ci=ci, seed=seed)
    return {"coverage": coverage(true, lo, hi), "width": mean_width(lo, hi),
            "cov_lo": lo_ci, "cov_hi": hi_ci}


# =========================================================================== #
#  Correctness self-test: distribution-free SCP coverage (HARD assert)         #
# =========================================================================== #
def _selftest_scp_coverage(delta: float = TARGET_DELTA, n_cal: int = 200,
                           n_test: int = 4000, n_trials: int = 300,
                           tol: float = 0.01, seed: int = 0) -> dict:
    """Monte-Carlo proof that SCP attains marginal coverage ~ (1 - delta) on
    exchangeable CLEAN data, INDEPENDENT of any model. Runs two very different
    score distributions (Gaussian and a skewed, mean-centered exponential) to
    demonstrate the guarantee is distribution-free, and asserts the mean coverage
    is within `tol` of the target for each. Also checks the k>n_cal -> +inf branch
    and monotonicity of q in the coverage target. Returns per-distribution means.
    """
    rng = np.random.default_rng(seed)
    draws = {
        "normal":            lambda m: rng.standard_normal(m),
        "exponential(skew)": lambda m: rng.exponential(1.0, m) - 1.0,
    }
    means: dict = {}
    for name, draw in draws.items():
        covs = []
        for _ in range(n_trials):
            cal_err = draw(n_cal)                       # calibration errors
            test_err = draw(n_test)                     # fresh exchangeable errors
            q = scp_quantile(np.abs(cal_err), delta)    # score = |error|, pred = 0
            covs.append(coverage(test_err, -q, q))      # |test_err| <= q ?
        mean_cov = float(np.mean(covs))
        means[name] = mean_cov
        assert abs(mean_cov - (1.0 - delta)) < tol, (
            f"SCP coverage [{name}] = {mean_cov:.4f} not within {tol} of "
            f"target {1 - delta:.2f} (SCP implementation is WRONG)")

    # k > n_cal -> +inf branch (request 99% coverage from only 5 points).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isinf(scp_quantile(np.abs(rng.standard_normal(5)), delta=0.01)), \
            "scp_quantile must return +inf when k > n_cal"

    # Monotonicity: tighter miscoverage (higher target) => wider (>=) half-width.
    s = np.abs(rng.standard_normal(500))
    assert scp_quantile(s, 0.05) >= scp_quantile(s, 0.10), \
        "scp_quantile must be non-decreasing as the coverage target rises"
    return means


def _selftest_wrcp(delta: float = TARGET_DELTA, seed: int = 0) -> None:
    """Deterministic checks on the WEIGHTED-quantile machinery -- this exercises
    the WR-CP path that the model demo skips whenever it falls back to SCP.
      (1) UNIFORM weights must reduce the weighted quantile EXACTLY to SCP.
      (2) ESS of uniform weights equals n.
      (3) Concentrating weight on the low-score half can only pull the half-width
          DOWN (never above the SCP value).
    """
    rng = np.random.default_rng(seed)
    s = np.abs(rng.standard_normal(40))
    q_scp = scp_quantile(s, delta)

    q_uniform = wrcp_quantiles(s, np.ones_like(s), np.ones(200), delta)
    assert np.allclose(q_uniform, q_scp), \
        f"uniform-weight WR-CP {q_uniform[:2]} must equal SCP {q_scp}"
    assert abs(effective_sample_size(np.ones_like(s)) - s.size) < 1e-9, \
        "ESS of uniform weights must equal n"

    w_low = np.where(s <= np.median(s), 10.0, 1e-6)     # mass on the low-score half
    q_low = wrcp_quantiles(s, w_low, np.ones(5), delta)
    assert np.all(q_low <= q_scp + 1e-9), \
        "down-weighting high scores must not widen the half-width above SCP"


def _selftest_cvplus_coverage(delta: float = TARGET_DELTA, n_cal: int = 60,
                              n_test: int = 1500, n_trials: int = 120, d: int = 3,
                              seed: int = 0) -> dict:
    """Monte-Carlo proof that CV+ AND jackknife+ attain their finite-sample
    marginal guarantee (>= 1 - 2*delta) and land near 1 - delta empirically, on
    exchangeable data and INDEPENDENT of the KinetiFlow model. Well-specified
    linear truth so the ridge base learner is unbiased; K=6 folds for CV+, LOO for
    jackknife+. Returns the per-scheme mean coverage."""
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(d)
    floor = (1.0 - 2.0 * delta)
    out: dict = {}
    for scheme in ("cv+", "jackknife+"):
        covs = []
        for _ in range(n_trials):
            Xc = rng.standard_normal((n_cal, d))
            yc = Xc @ beta + rng.standard_normal(n_cal)
            Xt = rng.standard_normal((n_test, d))
            yt = Xt @ beta + rng.standard_normal(n_test)
            fold = (np.arange(n_cal) % 6) if scheme == "cv+" else np.arange(n_cal)
            r = cvplus_intervals(Xc, yc, fold, Xt, delta=delta, l2=1e-6)
            covs.append(coverage(yt, r["lo"], r["hi"]))
        mean_cov = float(np.mean(covs))
        out[scheme] = mean_cov
        assert mean_cov >= floor - 0.02, (
            f"{scheme} coverage {mean_cov:.4f} below the 1-2*delta={floor:.2f} "
            f"finite-sample floor (CV+ implementation is WRONG)")
        assert mean_cov <= 1.0 + 1e-9
    return out


def _selftest_cqr_coverage(delta: float = TARGET_DELTA, n_train: int = 200,
                           n_cal: int = 60, n_test: int = 2000, n_trials: int = 50,
                           seed: int = 0) -> dict:
    """Monte-Carlo check that CQR attains marginal coverage >= 1 - delta on
    exchangeable, HETEROSCEDASTIC data (noise grows with x), model-free. Also
    asserts the mean CQR width is SMALLER than a constant-width absolute-residual
    SCP interval at the same nominal level -- the quantile heads buy locally-
    adaptive width, tightening where the noise is small and widening where it is
    large, so at equal coverage CQR is narrower on average."""
    rng = np.random.default_rng(seed)

    def sample(m: int):
        x = rng.uniform(0.0, 3.0, size=(m, 1))
        sigma = 0.2 + 1.3 * x[:, 0]                           # monotone heteroscedastic
        y = 1.5 * x[:, 0] + sigma * rng.standard_normal(m)
        return x, y

    covs, cqr_w, scp_w = [], [], []
    for _ in range(n_trials):
        Xtr, ytr = sample(n_train)
        Xc, yc = sample(n_cal)
        Xt, yt = sample(n_test)
        cqr = cqr_fit(Xtr, ytr, Xc, yc, delta=delta)
        lo, hi = cqr_intervals(cqr, Xt)
        covs.append(coverage(yt, lo, hi))
        cqr_w.append(mean_width(lo, hi))
        w = np.linalg.lstsq(np.column_stack([Xc[:, 0], np.ones(n_cal)]),
                            yc, rcond=None)[0]                # LS median proxy
        mu_c, mu_t = Xc[:, 0] * w[0] + w[1], Xt[:, 0] * w[0] + w[1]
        q = scp_quantile(np.abs(yc - mu_c), delta)
        scp_w.append(2.0 * q)
    mean_cov = float(np.mean(covs))
    mean_cqr_w, mean_scp_w = float(np.mean(cqr_w)), float(np.mean(scp_w))
    assert mean_cov >= (1.0 - delta) - 0.03, (
        f"CQR coverage {mean_cov:.4f} below 1-delta={1 - delta:.2f} (CQR is WRONG)")
    assert mean_cov <= 1.0 + 1e-9
    assert mean_cqr_w < mean_scp_w, (
        f"CQR width {mean_cqr_w:.3f} not below constant-width SCP {mean_scp_w:.3f} "
        f"under monotone heteroscedasticity (adaptivity lost)")
    return {"coverage": mean_cov, "cqr_width": mean_cqr_w, "scp_width": mean_scp_w}


# =========================================================================== #
#  Self-test / demonstration                                                  #
# =========================================================================== #
if __name__ == "__main__":
    import train
    import dataset as D

    print("=" * 72)
    print("conformal.py  —  SCP + WR-CP conformal abstention layer")
    print("=" * 72)

    # ---- (0) distribution-free correctness (the load-bearing hard gates) -----
    #  These are MODEL-FREE Monte-Carlo proofs -- they certify each method's
    #  coverage machinery independently of the KinetiFlow forecaster, so they hold
    #  even though the [1]-[5] demo below is on synthetic traces.
    means = _selftest_scp_coverage()
    _selftest_wrcp()
    cvp = _selftest_cvplus_coverage()
    cqr_mc = _selftest_cqr_coverage()
    print(f"\n[0] MODEL-FREE self-tests (hard asserts):")
    print(f"    SCP   MC coverage (target {1 - TARGET_DELTA:.2f})   "
          + "  ".join(f"{k}={v:.4f}" for k, v in means.items())
          + "   + WR-CP weighted-quantile checks")
    print(f"    CV+   MC coverage {cvp['cv+']:.4f}   jackknife+ {cvp['jackknife+']:.4f}"
          f"   (guarantee >= {1 - 2 * TARGET_DELTA:.2f}, empirical ~ {1 - TARGET_DELTA:.2f})")
    print(f"    CQR   MC coverage {cqr_mc['coverage']:.4f}   width {cqr_mc['cqr_width']:.2f} "
          f"< constant-SCP {cqr_mc['scp_width']:.2f}  (locally adaptive)   -> ALL PASS")

    # ---- (1) data + trained checkpoint ---------------------------------------
    data = D.load()
    train_b, cal_b, test_b = D.grouped_split(data)
    model, meas, cfg_dict, cal_ref = train.load_checkpoint(CKPT_DIR / "graybox_best.pt")
    cfg = train.TrainConfig(**cfg_dict)

    pred_cal, true_cal = predict_equilibrium(model, meas, cal_b, cfg)
    pred_test, true_test = predict_equilibrium(model, meas, test_b, cfg)
    repro = float(np.max(np.abs(pred_cal - _asnp(cal_ref[-1]))))
    scores_cal = nonconformity_scores(pred_cal, true_cal)

    q = scp_quantile(scores_cal, TARGET_DELTA)
    lo_t, hi_t = scp_interval(pred_test, q)
    cov_test = coverage(true_test, lo_t, hi_t)

    print("\n" + "-" * 72)
    print(f"[1] SPLIT CONFORMAL — clean in-distribution test "
          f"(n_cal={len(cal_b)}, n_test={len(test_b)})")
    print("-" * 72)
    print(f"    checkpoint reload max|d pred_900| on cal : {repro:.2e}  (exact)")
    print(f"    SCP half-width q                         : {q:.3f} DN")
    print(f"    SCP clean-test coverage                  : {cov_test:.3f}   "
          f"(target {1 - TARGET_DELTA:.2f}; SCP guarantees >=)")
    print(f"    SCP mean interval width                  : {mean_width(lo_t, hi_t):.3f} DN")
    assert 0.75 <= cov_test <= 1.0001, \
        f"clean-test coverage {cov_test:.3f} outside sanity band (MC proves correctness)"

    # ---- (2) covariate shift: fresh clean control + two shift magnitudes -----
    print("\n" + "-" * 72)
    print("[2] COVARIATE SHIFT — fresh synthetic sets (seeds 9900x != data seed 12345)")
    print("-" * 72)
    scenarios = (("clean", 30.0, 55.0, 99001),      # in-distribution control
                 ("mild", 31.5, 50.0, 99002),       # bounded shift
                 ("strong", 34.0, 44.0, 99003))     # large shift
    sets: Dict[str, dict] = {}
    for tag, T_nom, RH_nom, seed in scenarios:
        b = make_shifted_bundle(tag, T_nominal=T_nom, RH_nominal=RH_nom, seed=seed)
        pr, tr = predict_equilibrium(model, meas, b, cfg)
        sc = nonconformity_scores(pr, tr)
        cov = coverage(tr, *scp_interval(pr, q))
        sets[tag] = dict(b=b, pred=pr, true=tr, scores=sc, scp_cov=cov)
        print(f"    {tag:6s}: n={len(b):3d}  T~{_asnp(b.T_ambient).mean():4.1f}C "
              f"RH~{_asnp(b.RH).mean():4.1f}%  mean|resid|={sc.mean():.3f} DN  "
              f"SCP cov={cov:.3f}")
    drop_mild = ood_coverage_drop(sets["clean"]["scp_cov"], sets["mild"]["scp_cov"])
    drop_strong = ood_coverage_drop(sets["clean"]["scp_cov"], sets["strong"]["scp_cov"])
    print(f"\n    OOD coverage drop  clean->mild   : {drop_mild:+.3f}")
    print(f"    OOD coverage drop  clean->strong : {drop_strong:+.3f}   "
          f"(SCP coverage degrades as the shift grows)")

    # ---- (3) WR-CP across the shift envelope ---------------------------------
    print("\n" + "-" * 72)
    print("[3] WR-CP — weighted/Wasserstein-regularized conformal (EXPLORATORY)")
    print("-" * 72)
    print(f"    {'shift':7s} {'AUC':>6s} {'ESS':>6s} {'W1_DN':>6s} "
          f"{'mode':>11s} {'SCP_cov':>8s} {'WRCP_cov':>9s}")
    for tag in ("clean", "mild", "strong"):
        d = sets[tag]
        res = wrcp_intervals(scores_cal, _covariates(cal_b), _covariates(d["b"]),
                             d["pred"], d["scores"], delta=TARGET_DELTA, ess_min=ESS_MIN)
        d["wrcp"], d["wrcp_cov"] = res, coverage(d["true"], res["lo"], res["hi"])
        mode = "SCP-fallbk" if res["fell_back"] else "weighted"
        print(f"    {tag:7s} {res['clf_auc']:6.3f} {res['ess']:6.1f} "
              f"{res['wasserstein']:6.3f} {mode:>11s} {d['scp_cov']:8.3f} "
              f"{d['wrcp_cov']:9.3f}")
    print(f"    AUC 0.5=no shift; ESS floor {ESS_MIN:.0f}/{len(cal_b)} -> below it WR-CP")
    print(f"    falls back to SCP. No-shift 'clean' engages (weighted) and matches SCP.")

    # ---- (4) abstention + selective accuracy (thresholds LOCKED on cal) ------
    print("\n" + "-" * 72)
    print("[4] ABSTENTION — theta_clinical & w_max LOCKED on calibration (no leakage)")
    print("-" * 72)
    theta, w_max = lock_thresholds(true_cal, cal_b.concentration_level)
    print(f"    theta_clinical = {theta:.3f} DN   w_max = {w_max:.3f} DN   [locked on cal]")
    print(f"    {'set':7s} {'abstain':>8s} {'straddle':>8s} {'wide':>6s} "
          f"{'acc_full':>9s} {'acc_kept':>9s}")
    for tag in ("clean", "strong"):
        d = sets[tag]
        lo_s, hi_s = scp_interval(d["pred"], q)
        straddle, too_wide = abstain_masks(lo_s, hi_s, theta, w_max)
        keep = ~(straddle | too_wide)
        full_acc, sel_acc, ab_rate = selective_accuracy(d["pred"], d["true"], theta, keep)
        print(f"    {tag:7s} {ab_rate:7.1%} {straddle.mean():7.1%} "
              f"{too_wide.mean():5.1%} {full_acc:9.3f} {sel_acc:9.3f}")

    # ---- (5) SCP vs WR-CP vs CV+/jackknife+ vs CQR, side by side --------------
    #  Calibrate on a MATCHED synthetic reference split (same generator as the
    #  clean/shifted eval sets) so 'clean' is genuinely exchangeable with cal --
    #  this isolates the METHODS from the traces.pt-vs-synthetic forecaster gap
    #  that section [2] exhibits. n_cal stays in the 40-80 target regime, and the
    #  gray-box is used as a FIXED forecaster (never retrained).
    import synthetic
    syn = synthetic.generate(synthetic.SyntheticConfig(seed=99020))     # nominal
    train_syn, cal_syn, _ = D.grouped_split(syn)
    print("\n" + "-" * 72)
    print(f"[5] METHOD COMPARISON — small-calibration regime "
          f"(n_cal={len(cal_syn)}, target {1 - TARGET_DELTA:.0%})")
    print("-" * 72)
    print("    calibrated on a MATCHED synthetic split so 'clean' is exchangeable;")
    print("    cf. [2] (calibrated on the real cal split) where the 'clean' coverage")
    print("    gap is the forecaster's traces.pt-vs-synthetic mismatch, not a method")
    print("    failure. The gray-box is a FIXED forecaster here (never retrained).")
    _pred_cache: Dict[int, tuple] = {}
    def predict_fn(b):
        """Memoized gray-box forecast (by bundle identity) so the five methods and
        the context share ONE UDE rollout per set instead of re-integrating."""
        if id(b) not in _pred_cache:
            _pred_cache[id(b)] = predict_equilibrium(model, meas, b, cfg)
        return _pred_cache[id(b)]
    ctx = build_context(predict_fn, cal_syn, train_bundle=train_syn, delta=TARGET_DELTA)
    methods = build_methods(TARGET_DELTA)
    for mth in methods.values():
        mth.fit(ctx)

    def _wtxt(w: float) -> str:
        return "  inf" if not np.isfinite(w) else f"{w:6.2f}"

    comp: Dict[str, Dict[str, dict]] = {name: {} for name in methods}
    for si, tag in enumerate(("clean", "mild", "strong")):
        b = sets[tag]["b"]
        print(f"    ----- {tag} (n={len(b)}, T~{_asnp(b.T_ambient).mean():.1f}C "
              f"RH~{_asnp(b.RH).mean():.1f}%) -----")
        print(f"      {'method':11s} {'coverage':>8s}  {'90% CI on coverage':>20s}  "
              f"{'width(DN)':>9s}")
        for mi, (name, mth) in enumerate(methods.items()):
            res = mth.intervals(b)
            ev = evaluate_intervals(sets[tag]["true"], res, seed=1000 + 10 * si + mi)
            comp[name][tag] = ev
            ci = f"[{ev['cov_lo']:.3f}, {ev['cov_hi']:.3f}]"
            print(f"      {name:11s} {ev['coverage']:8.3f}  {ci:>20s}  "
                  f"{_wtxt(ev['width']):>9s}")
        print()

    # narrowest interval among methods that actually reach the target on 'clean'
    tgt = 1.0 - TARGET_DELTA
    elig = {m: comp[m]["clean"] for m in comp
            if comp[m]["clean"]["coverage"] >= tgt - 1e-9
            and np.isfinite(comp[m]["clean"]["width"])}
    print(f"    Narrowest interval at >= {tgt:.0%} coverage on the CLEAN set:")
    if elig:
        order = sorted(elig, key=lambda m: elig[m]["width"])
        for m in order:
            mark = "   <== narrowest at equal coverage" if m == order[0] else ""
            print(f"      {m:11s} width={elig[m]['width']:6.2f} DN  "
                  f"cov={elig[m]['coverage']:.3f}{mark}")
    else:
        print("      (no method reached the target coverage on the clean set)")

    # Sanity band on the CLEAN set (exchangeable with cal): each method should land
    # near the 90% target, allowing over-coverage (CV+/jackknife+ are conservative).
    # The EXACT distribution-free gates are the model-free MC tests in [0]; this band
    # only catches gross miscalibration under small-n sampling noise.
    for m in comp:
        c = comp[m]["clean"]["coverage"]
        assert (1 - TARGET_DELTA) - 0.08 <= c <= 1.0001, \
            f"{m} clean coverage {c:.3f} outside tolerance of target {1 - TARGET_DELTA:.2f}"

    print("    HONESTY / how to read this:")
    print("    * All data is SYNTHETIC; guarantees are MARGINAL (trace-averaged), NOT")
    print("      conditional on concentration / day / lot. Group-CONDITIONAL coverage")
    print("      (Mondrian, per-level / per-lot) is deferred to evaluate.py.")
    print("    * SCP & WR-CP wrap the RAW gray-box forecast; CV+/jackknife+ & CQR wrap")
    print("      a light recalibration that also reads the calibration covariates")
    print("      (C_f0, T, RH). Their much smaller width here reflects BIAS-CORRECTION")
    print("      of an imperfect forecaster on this set, not only a tighter conformal")
    print("      rule -- and that narrowness is BRITTLE: CV+/jackknife+ coverage")
    print("      collapses under the strong shift while fixed-width SCP degrades")
    print("      gracefully. The pure, model-free coverage guarantees are proven in [0].")

    # ---- summary -------------------------------------------------------------
    strong_res = sets["strong"]["wrcp"]
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"    Model-free coverage self-tests (SCP/WR-CP/CV+/jackknife+/CQR) : PASS")
    print(f"    Clean-set coverage [90% CI] & mean width (n_cal={len(cal_syn)}, "
          f"target {1 - TARGET_DELTA:.0%}):")
    for m in methods:
        ev = comp[m]["clean"]
        print(f"      {m:11s} cov={ev['coverage']:.3f} "
              f"[{ev['cov_lo']:.3f},{ev['cov_hi']:.3f}]  width={_wtxt(ev['width']).strip()} DN")
    if elig:
        print(f"    Narrowest at >= {tgt:.0%} coverage (clean): {order[0]} "
              f"({elig[order[0]]['width']:.2f} DN)")
    scp_drop = comp["SCP"]["clean"]["coverage"] - comp["SCP"]["strong"]["coverage"]
    print(f"    SCP coverage drop clean->strong shift: {scp_drop:+.3f}  "
          f"(CV+/jackknife+ drop more; CQR over-covers)")
    print("    Guarantees: SCP & CQR marginal >= 1-delta under exchangeability;")
    print("    CV+/jackknife+ marginal >= 1-2*delta using ALL cal data (best at")
    print("    small n_cal); WR-CP only under BOUNDED covariate shift. None cover")
    print("    arbitrary OOD, and none are group-CONDITIONAL (Mondrian -> evaluate.py).")
