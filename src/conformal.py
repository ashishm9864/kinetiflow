"""Leakage-resistant split and importance-weighted conformal prediction.

The old module was invalid: model selection reused conformal calibration rows,
its purported WR-CP routine read hidden test residuals, and its density ratio
used true concentration.  This module makes those operations impossible through
immutable calibrators whose interval methods accept predictions and deployable
covariates only.

``SCP`` has the standard finite-sample marginal guarantee only when calibration
and test examples are exchangeable and the fitted point model is independent of
the calibration outcomes.  Group-held-out days/lots with hierarchical effects do
not automatically satisfy that assumption; repeated group assignments are an
empirical sensitivity analysis, not a theorem.

The weighted implementation is correctly labeled ``IW-CP``.  It is the
prediction-phase primitive used by Xu et al. WR-CP, but it is not itself their
training-time Wasserstein representation regularizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from interfaces import ForecastInputs, as_numpy


TARGET_DELTA = 0.10


def _vector(value, name: str, *, finite: bool = True) -> np.ndarray:
    result = as_numpy(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}")
    if result.size == 0:
        raise ValueError(f"{name} is empty")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def nonconformity_scores(pred, true) -> np.ndarray:
    prediction = _vector(pred, "pred")
    target = _vector(true, "true")
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    return np.abs(target - prediction)


def scp_quantile(scores, delta: float = TARGET_DELTA) -> float:
    """Exact finite-sample split-conformal order statistic.

    ``k = ceil((n+1)(1-delta))``.  If ``k>n``, the only honest interval is
    unbounded; the order is never silently clipped.
    """
    values = np.sort(_vector(scores, "calibration scores"))
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    n = values.size
    k = math.ceil((n + 1) * (1.0 - delta))
    if k > n:
        minimum = math.ceil((1.0 - delta) / delta)
        warnings.warn(
            f"n_cal={n} cannot certify {1-delta:.1%} coverage; need at least "
            f"{minimum}, returning an infinite interval",
            stacklevel=2,
        )
        return math.inf
    return float(values[k - 1])


@dataclass(frozen=True)
class SCPCalibrator:
    q: float
    delta: float
    n_cal: int
    provenance: str

    def interval(self, prediction) -> Tuple[np.ndarray, np.ndarray]:
        pred = _vector(prediction, "prediction")
        return pred - self.q, pred + self.q


def fit_scp(
    calibration_prediction,
    calibration_target,
    delta: float = TARGET_DELTA,
    provenance: str = "calibration",
) -> SCPCalibrator:
    scores = nonconformity_scores(calibration_prediction, calibration_target)
    return SCPCalibrator(scp_quantile(scores, delta), delta, len(scores), provenance)


def scp_interval(prediction, q: float) -> Tuple[np.ndarray, np.ndarray]:
    pred = _vector(prediction, "prediction")
    return pred - q, pred + q


def effective_sample_size(weights) -> float:
    w = _vector(weights, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    denominator = float(np.dot(w, w))
    return float(w.sum() ** 2 / denominator) if denominator > 0 else 0.0


def weighted_quantiles(cal_scores, cal_weights, test_weights, delta: float = TARGET_DELTA) -> np.ndarray:
    """Tibshirani et al. weighted split-conformal quantiles with +inf atom."""
    scores = _vector(cal_scores, "calibration scores")
    wc = _vector(cal_weights, "calibration weights")
    wt = _vector(test_weights, "test weights")
    if scores.shape != wc.shape:
        raise ValueError("calibration score/weight shape mismatch")
    if np.any(wc < 0) or np.any(wt < 0):
        raise ValueError("importance weights must be nonnegative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    prefix = np.cumsum(wc[order])
    if prefix[-1] <= 0:
        return np.full(wt.shape, np.inf)
    target_mass = (1.0 - delta) * (prefix[-1] + wt)
    index = np.searchsorted(prefix, target_mass, side="left")
    q = np.full(wt.shape, np.inf, dtype=np.float64)
    valid = index < len(sorted_scores)
    q[valid] = sorted_scores[index[valid]]
    return q


def deployable_covariates(inputs: ForecastInputs) -> np.ndarray:
    """T/RH plus early-trace summaries; no concentration or target metadata."""
    intensity = as_numpy(inputs.I_early)
    t = as_numpy(inputs.t)
    duration = max(float(t[-1] - t[0]), 1e-9)
    slope = (intensity[:, -1] - intensity[:, 0]) / duration
    return np.column_stack([
        as_numpy(inputs.T_ambient),
        as_numpy(inputs.RH),
        intensity[:, 0],
        intensity[:, -1],
        intensity.mean(axis=1),
        intensity.std(axis=1),
        slope,
    ])


@dataclass
class DensityRatioModel:
    """Likelihood-ratio estimator fit on calibration and an independent target-reference pool."""

    scaler: object
    classifier: object
    n_source: int
    n_target_reference: int
    cv_auc: float

    def weights(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        probability = self.classifier.predict_proba(self.scaler.transform(x))[:, 1]
        probability = np.clip(probability, 1e-4, 1.0 - 1e-4)
        return (probability / (1.0 - probability)) * (self.n_source / self.n_target_reference)


def fit_density_ratio(source_features, target_reference_features, seed: int = 0) -> DensityRatioModel:
    """Fit without using final test rows or any outcome."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    source = np.asarray(source_features, dtype=np.float64)
    target = np.asarray(target_reference_features, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("density-ratio features must be 2-D with matching columns")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("non-finite density-ratio feature")
    x = np.vstack([source, target])
    y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    smallest = int(min(np.bincount(y.astype(int))))
    folds = min(5, smallest)
    if folds >= 2:
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        probability = cross_val_predict(classifier, xs, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, probability))
    else:
        auc = float("nan")
    classifier.fit(xs, y)
    return DensityRatioModel(scaler, classifier, len(source), len(target), auc)


@dataclass(frozen=True)
class IWCPCalibrator:
    scores: np.ndarray
    calibration_weights: np.ndarray
    delta: float
    density_ratio: DensityRatioModel
    provenance: str

    @property
    def ess(self) -> float:
        return effective_sample_size(self.calibration_weights)

    @property
    def ess_fraction(self) -> float:
        return self.ess / len(self.calibration_weights)

    def interval(self, prediction, deployable_features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = _vector(prediction, "prediction")
        test_weights = self.density_ratio.weights(deployable_features)
        if len(test_weights) != len(pred):
            raise ValueError("prediction/test-feature shape mismatch")
        q = weighted_quantiles(self.scores, self.calibration_weights, test_weights, self.delta)
        return pred - q, pred + q, q


def fit_iwcp(
    calibration_prediction,
    calibration_target,
    calibration_features,
    density_ratio: DensityRatioModel,
    delta: float = TARGET_DELTA,
    provenance: str = "calibration + independent target reference",
) -> IWCPCalibrator:
    scores = nonconformity_scores(calibration_prediction, calibration_target)
    features = np.asarray(calibration_features, dtype=np.float64)
    if len(features) != len(scores):
        raise ValueError("calibration feature/score shape mismatch")
    weights = density_ratio.weights(features)
    return IWCPCalibrator(scores.copy(), weights.copy(), delta, density_ratio, provenance)


def coverage(true, lo, hi) -> float:
    target = _vector(true, "true")
    lower = _vector(lo, "lower", finite=False)
    upper = _vector(hi, "upper", finite=False)
    if target.shape != lower.shape or target.shape != upper.shape:
        raise ValueError("coverage shape mismatch")
    return float(np.mean((target >= lower) & (target <= upper)))


def mean_width(lo, hi) -> float:
    lower = _vector(lo, "lower", finite=False)
    upper = _vector(hi, "upper", finite=False)
    if lower.shape != upper.shape:
        raise ValueError("interval shape mismatch")
    return float(np.mean(upper - lower))


@dataclass(frozen=True)
class AbstentionThresholds:
    theta_clinical: float
    w_max: float
    provenance: str = "calibration"


def lock_thresholds(
    calibration_target,
    calibration_levels,
    negative_level: float = 0.5,
    positive_level: float = 1.0,
) -> AbstentionThresholds:
    target = _vector(calibration_target, "calibration target")
    levels = _vector(calibration_levels, "calibration levels")
    if target.shape != levels.shape:
        raise ValueError("calibration target/level mismatch")
    negative = target[levels == negative_level]
    positive = target[levels == positive_level]
    if not len(negative) or not len(positive):
        raise ValueError("calibration lacks prespecified negative/positive levels")
    theta = 0.5 * (float(negative.mean()) + float(positive.mean()))
    width = abs(float(positive.mean()) - float(negative.mean()))
    return AbstentionThresholds(theta, width)


def abstain(lo, hi, thresholds: AbstentionThresholds) -> np.ndarray:
    lower = _vector(lo, "lower", finite=False)
    upper = _vector(hi, "upper", finite=False)
    if lower.shape != upper.shape:
        raise ValueError("interval shape mismatch")
    width = upper - lower
    straddles = (lower <= thresholds.theta_clinical) & (upper >= thresholds.theta_clinical)
    invalid = ~np.isfinite(width)
    return straddles | (width > thresholds.w_max) | invalid


def selective_accuracy(pred, true, thresholds: AbstentionThresholds, abstained) -> Tuple[float, float, float]:
    prediction = _vector(pred, "prediction")
    target = _vector(true, "true")
    rejected = np.asarray(abstained, dtype=bool)
    if prediction.shape != target.shape or rejected.shape != prediction.shape:
        raise ValueError("selective metric shape mismatch")
    correct = (prediction >= thresholds.theta_clinical) == (target >= thresholds.theta_clinical)
    kept = ~rejected
    return (
        float(correct.mean()),
        float(correct[kept].mean()) if kept.any() else float("nan"),
        float(rejected.mean()),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isinf(scp_quantile(np.arange(8.0), 0.1))
    assert scp_quantile(np.arange(9.0), 0.1) == 8.0
    scores = np.abs(rng.normal(size=40))
    q = scp_quantile(scores)
    weighted = weighted_quantiles(scores, np.ones(40), np.ones(5))
    assert np.allclose(weighted, q)
    calibrator = fit_scp(np.zeros(40), scores)
    lo, hi = calibrator.interval(np.zeros(10))
    assert np.allclose(hi - lo, 2 * calibrator.q)
    print("FINITE-SAMPLE SCP + LEAKAGE-CLOSED API: PASS")
