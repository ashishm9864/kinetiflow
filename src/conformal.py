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
    * WR-CP (weighted, for covariate shift) is added in Part 2; it restores
      coverage only under BOUNDED covariate shift, and is exploratory at n_cal~40.
    Neither method covers arbitrary OOD.

LEAKAGE
    Conformal calibration uses the CALIBRATION split only (leakage-safe by day and
    lot, from dataset.py). Test / OOD data never enter calibration.

Run the self-test:  python src/conformal.py
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"

TARGET_DELTA = 0.10          # default miscoverage -> 90% prediction intervals


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


# --------------------------------------------------------------------------- #
#  Part 1 — SPLIT CONFORMAL PREDICTION (SCP)                                   #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Model-prediction helper (imports train.py; does not modify it)             #
# --------------------------------------------------------------------------- #
def predict_equilibrium(model, meas, bundle, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast the 15-min equilibrium intensity for a TraceBundle.

    Returns (pred_900, true_900) as numpy arrays: the model's I(900 s) forecast
    (last timepoint of the no-grad rollout) and the noise-free ground truth.
    Uses train.eval_loss, which handles per-trace covariates internally.
    """
    import train
    _, _, I_pred = train.eval_loss(model, meas, bundle, cfg)   # [T, n]
    pred_900 = _asnp(I_pred[-1])                                # [n]
    true_900 = _asnp(bundle.true_I_900)                        # [n]
    return pred_900, true_900


# --------------------------------------------------------------------------- #
#  Correctness self-test: distribution-free SCP coverage (HARD assert)         #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Self-test / demonstration                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 70)
    print("conformal.py  —  Part 1: Split Conformal Prediction (SCP)")
    print("=" * 70)

    # ---- 1) distribution-free coverage proof (the load-bearing hard assert) --
    means = _selftest_scp_coverage()
    print("\n[MC self-test] SCP marginal coverage on clean exchangeable data "
          f"(target {1 - TARGET_DELTA:.2f}):")
    for name, m in means.items():
        print(f"    {name:20s}: {m:.4f}")
    print("    k>n_cal -> +inf, monotonicity: OK")
    print("    SCP DISTRIBUTION-FREE COVERAGE: PASS")

    # ---- 2) real model: calibrate on CAL, cover the clean TEST split ---------
    import train
    import dataset as D

    data = D.load()
    _, cal_b, test_b = D.grouped_split(data)
    model, meas, cfg_dict, cal_ref = train.load_checkpoint(CKPT_DIR / "graybox_best.pt")
    cfg = train.TrainConfig(**cfg_dict)

    pred_cal, true_cal = predict_equilibrium(model, meas, cal_b, cfg)
    pred_test, true_test = predict_equilibrium(model, meas, test_b, cfg)

    # sanity: our recomputed cal forecast matches the checkpoint's stored reference
    repro = float(np.max(np.abs(pred_cal - _asnp(cal_ref[-1]))))

    scores_cal = nonconformity_scores(pred_cal, true_cal)
    q = scp_quantile(scores_cal, TARGET_DELTA)
    lo, hi = scp_interval(pred_test, q)
    cov = coverage(true_test, lo, hi)
    width = mean_width(lo, hi)

    print("\n" + "-" * 70)
    print("real gray-box checkpoint  (calibrate on CAL, evaluate clean TEST)")
    print("-" * 70)
    print(f"n_cal={len(cal_b)}  n_test={len(test_b)}  target coverage="
          f"{1 - TARGET_DELTA:.2f}")
    print(f"checkpoint reload max|d pred_900| on cal = {repro:.2e} "
          f"{'PASS' if repro < 1e-4 else 'WARN'} (<1e-4)")
    print(f"SCP half-width q          = {q:.3f} DN")
    print(f"SCP clean-test coverage   = {cov:.3f}")
    print(f"SCP mean interval width   = {width:.3f} DN")

    # n_test=40 => finite-sample sampling noise on the coverage estimate; the
    # distribution-free CORRECTNESS is asserted by the Monte-Carlo test above.
    # Here we require the realized coverage to sit in a generous binomial band.
    lo_band = 0.75
    ok = lo_band <= cov <= 1.0001
    print(f"clean-test coverage in [{lo_band:.2f}, 1.00] band: "
          f"{'PASS' if ok else 'FAIL'}  (n={len(test_b)} sampling noise; "
          f"hard correctness = MC above)")
    assert ok, f"clean-test coverage {cov:.3f} outside sanity band [{lo_band}, 1.0]"

    print("\nSCP (Part 1): OK")
