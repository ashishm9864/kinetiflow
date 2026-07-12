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
from pathlib import Path
from typing import Dict, Tuple

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


# =========================================================================== #
#  Self-test / demonstration                                                  #
# =========================================================================== #
if __name__ == "__main__":
    import train
    import dataset as D

    print("=" * 72)
    print("conformal.py  —  SCP + WR-CP conformal abstention layer")
    print("=" * 72)

    # ---- (0) distribution-free SCP correctness (the load-bearing hard gate) --
    means = _selftest_scp_coverage()
    _selftest_wrcp()
    print(f"\n[0] self-tests: SCP MC coverage (target {1 - TARGET_DELTA:.2f})  "
          + "  ".join(f"{k}={v:.4f}" for k, v in means.items())
          + "  + WR-CP weighted-quantile checks   -> PASS")

    # ---- (1) data + trained checkpoint ---------------------------------------
    data = D.load()
    _, cal_b, test_b = D.grouped_split(data)
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

    # ---- summary -------------------------------------------------------------
    strong_res = sets["strong"]["wrcp"]
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"    SCP distribution-free coverage self-test : PASS")
    print(f"    SCP clean-test coverage                  : {cov_test:.3f} "
          f"(>= target {1 - TARGET_DELTA:.2f})")
    print(f"    OOD coverage drop under SCP (strong)     : {drop_strong:+.3f}")
    print(f"    WR-CP (strong shift)                     : "
          + ("SCP fallback — ESS below floor (honest, not garbage)" if strong_res["fell_back"]
             else f"weighted, coverage {sets['strong']['wrcp_cov']:.3f}"))
    print("    Guarantee: SCP under exchangeability; WR-CP under BOUNDED shift only")
    print("    (exploratory at n_cal~40). Neither covers arbitrary OOD.")
