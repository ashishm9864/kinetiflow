"""Model-agnostic end-to-end evaluation for KinetiFlow-CP.

This module trains the physics-only, residual-only, gray-box, and capacity-
matched TCN forecasters on identical grouped roles; calibrates every interval on
the untouched conformal-calibration role; and writes machine-readable and
judge-facing results.

The primary comparison is ``graybox + SCP`` versus ``TCN + SCP``.  ``WR-CP``
rows combine the Xu-style multi-source Wasserstein training objective implemented
in the trainers with its importance-weighted prediction phase.  A separate
``IW-CP`` row on the unregularized model makes the contribution of training-time
Wasserstein regularization visible.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution

import conformal as C
import dataset as D
from interfaces import PointForecaster, as_numpy, inputs_from_bundle
import synthetic
import tcn_baseline as TCN
import train as GB


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
PREFIXES = (20.0, 30.0, 40.0, 50.0, 60.0)
TARGET_COVERAGE = 0.90


def rmse(prediction, target) -> float:
    """Root-mean-square endpoint error in DN."""
    p, y = as_numpy(prediction), as_numpy(target)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(prediction, target) -> float:
    """Mean absolute endpoint error in DN."""
    p, y = as_numpy(prediction), as_numpy(target)
    return float(np.mean(np.abs(p - y)))


def exact_coverage_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Clopper-Pearson interval for an observed coverage proportion."""
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta_distribution.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def selective_diagnostic_accuracy(
    prediction,
    true_levels,
    theta_clinical: float,
    abstained,
    positive_level: float = 1.0,
) -> Tuple[float, float, float, int]:
    """Clinical-call accuracy using known concentration class as ground truth.

    This avoids the old circular metric that thresholded both prediction and the
    optical target, and it exposes hook-effect false negatives when high-dose
    data are eventually collected.
    """
    pred = as_numpy(prediction)
    levels = as_numpy(true_levels)
    rejected = np.asarray(abstained, dtype=bool)
    predicted_positive = pred >= theta_clinical
    true_positive = levels >= positive_level
    correct = predicted_positive == true_positive
    kept = ~rejected
    return (
        float(correct.mean()),
        float(correct[kept].mean()) if kept.any() else float("nan"),
        float(rejected.mean()),
        int(kept.sum()),
    )


def decision_times(interval_widths: Mapping[float, np.ndarray], w_max: float) -> np.ndarray:
    """Earliest allowed prefix whose interval width is at most ``w_max``.

    Returns ``+inf`` for traces that never decide by 60 s.  Prefix-specific
    calibration is descriptive; ordinary SCP does not provide a simultaneous
    optional-stopping guarantee across all prefixes.
    """
    windows = sorted(interval_widths)
    n = len(np.asarray(interval_widths[windows[0]]))
    result = np.full(n, np.inf, dtype=np.float64)
    for window in windows:
        width = np.asarray(interval_widths[window], dtype=np.float64)
        eligible = np.isinf(result) & np.isfinite(width) & (width <= w_max)
        result[eligible] = window
    return result


def _bundle(name: str, data: Dict[str, torch.Tensor]) -> D.TraceBundle:
    return D._make_bundle(name, list(range(len(data["I_obs"]))), data)


def _external_sets() -> Dict[str, D.TraceBundle]:
    """Fresh, disjoint target-reference and held-out OOD sets."""
    clean_reference = synthetic.generate(synthetic.SyntheticConfig(
        n_groups=6, group_id_offset=1_000, seed=91_001,
    ))
    ood_reference = synthetic.generate(synthetic.SyntheticConfig(
        n_groups=6, group_id_offset=2_000, T_nominal=38.0, RH_nominal=75.0,
        seed=91_002,
    ))
    ood_test = synthetic.generate(synthetic.SyntheticConfig(
        n_groups=6, group_id_offset=3_000, T_nominal=38.0, RH_nominal=75.0,
        seed=91_003,
    ))
    return {
        "clean_reference": _bundle("clean_reference", clean_reference),
        "ood_reference": _bundle("ood_reference", ood_reference),
        "ood_test": _bundle("ood_test", ood_test),
    }


def _train_models(
    roles: Mapping[str, D.TraceBundle],
    split_seed: int,
    epochs: int,
    patience: int,
) -> Dict[str, PointForecaster]:
    models: Dict[str, PointForecaster] = {}
    for dynamics in ("physics_only", "residual_only", "graybox"):
        cfg = GB.TrainConfig(
            dynamics=dynamics,
            tag=dynamics,
            epochs=epochs,
            patience=patience,
            seed=0,
        )
        result = GB.train(roles["train"], roles["validation"], cfg)
        models[dynamics] = result["model"]
        if split_seed == 0:
            GB.save_checkpoint(result, CHECKPOINT_DIR / f"{dynamics}_best.pt")

    gray_wr_cfg = GB.TrainConfig(
        dynamics="graybox", tag="graybox_wr", epochs=epochs,
        patience=patience, seed=0, wr_beta=0.10,
    )
    gray_wr = GB.train(
        roles["train"], roles["validation"], gray_wr_cfg, roles["wr_reference"]
    )
    models["graybox_wr"] = gray_wr["model"]
    if split_seed == 0:
        GB.save_checkpoint(gray_wr, CHECKPOINT_DIR / "graybox_wr_best.pt")

    tcn_cfg = TCN.TCNConfig(epochs=max(400, epochs * 5), patience=max(60, patience * 4), seed=0)
    tcn = TCN.train_tcn(roles["train"], roles["validation"], tcn_cfg)
    models["tcn"] = tcn["model"]
    if split_seed == 0:
        TCN.save_checkpoint(tcn, CHECKPOINT_DIR / "tcn_baseline.pt")

    tcn_wr_cfg = TCN.TCNConfig(
        epochs=max(400, epochs * 5), patience=max(60, patience * 4), seed=0, wr_beta=0.10
    )
    tcn_wr = TCN.train_tcn(
        roles["train"], roles["validation"], tcn_wr_cfg, roles["wr_reference"]
    )
    models["tcn_wr"] = tcn_wr["model"]
    if split_seed == 0:
        TCN.save_checkpoint(tcn_wr, CHECKPOINT_DIR / "tcn_wr_best.pt")
    return models


def _predictions(
    model: PointForecaster,
    bundle: D.TraceBundle,
    prefixes: Sequence[float] = PREFIXES,
) -> Dict[float, np.ndarray]:
    return {window: model.predict(inputs_from_bundle(bundle, window)) for window in prefixes}


def _row(
    *,
    split_seed: int,
    model_name: str,
    conformal_method: str,
    scenario: str,
    prediction: np.ndarray,
    bundle: D.TraceBundle,
    lo: np.ndarray,
    hi: np.ndarray,
    thresholds: C.AbstentionThresholds,
    decision: np.ndarray,
    ess: float = float("nan"),
    ess_fraction: float = float("nan"),
    density_auc: float = float("nan"),
) -> Dict:
    target = as_numpy(bundle.true_I_900)
    covered = (target >= lo) & (target <= hi)
    successes = int(covered.sum())
    ci_low, ci_high = exact_coverage_interval(successes, len(target))
    rejected = C.abstain(lo, hi, thresholds)
    full_accuracy, selective, abstention_rate, n_kept = selective_diagnostic_accuracy(
        prediction, bundle.concentration_level, thresholds.theta_clinical, rejected
    )
    finite_decision = decision[np.isfinite(decision)]
    return {
        "record_type": "run",
        "split_seed": split_seed,
        "model": model_name,
        "conformal": conformal_method,
        "scenario": scenario,
        "n": len(bundle),
        "rmse_dn": rmse(prediction, target),
        "mae_dn": mae(prediction, target),
        "coverage": successes / len(target),
        "coverage_successes": successes,
        "coverage_ci95_low": ci_low,
        "coverage_ci95_high": ci_high,
        "mean_width_dn": C.mean_width(lo, hi),
        "abstention_rate": abstention_rate,
        "full_diagnostic_accuracy": full_accuracy,
        "selective_diagnostic_accuracy": selective,
        "n_non_abstained": n_kept,
        "decision_time_mean_s": float(finite_decision.mean()) if len(finite_decision) else float("inf"),
        "decision_time_median_s": float(np.median(finite_decision)) if len(finite_decision) else float("inf"),
        "decided_by_60_rate": float(np.isfinite(decision).mean()),
        "iw_ess": ess,
        "iw_ess_fraction": ess_fraction,
        "density_ratio_cv_auc": density_auc,
        "theta_clinical_dn": thresholds.theta_clinical,
        "w_max_dn": thresholds.w_max,
    }


def evaluate_split(
    data: Dict[str, torch.Tensor],
    external: Mapping[str, D.TraceBundle],
    split_seed: int,
    epochs: int,
    patience: int,
) -> List[Dict]:
    roles = D.grouped_role_split(data, split_seed)
    models = _train_models(roles, split_seed, epochs, patience)
    thresholds = C.lock_thresholds(
        roles["calibration"].true_I_900,
        roles["calibration"].concentration_level,
    )
    rows: List[Dict] = []

    for model_name, model in models.items():
        cal_pred = _predictions(model, roles["calibration"])
        clean_pred = _predictions(model, roles["test"])
        ood_pred = _predictions(model, external["ood_test"])
        scp_by_prefix: Dict[float, C.SCPCalibrator] = {
            window: C.fit_scp(
                cal_pred[window], roles["calibration"].true_I_900,
                provenance=f"split={split_seed}/calibration/prefix={window}",
            )
            for window in PREFIXES
        }
        for scenario, bundle, prediction_by_prefix in (
            ("clean", roles["test"], clean_pred),
            ("ood", external["ood_test"], ood_pred),
        ):
            widths: Dict[float, np.ndarray] = {}
            intervals: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            for window in PREFIXES:
                lo, hi = scp_by_prefix[window].interval(prediction_by_prefix[window])
                intervals[window] = (lo, hi)
                widths[window] = hi - lo
            decision = decision_times(widths, thresholds.w_max)
            lo, hi = intervals[60.0]
            rows.append(_row(
                split_seed=split_seed, model_name=model_name,
                conformal_method="SCP", scenario=scenario,
                prediction=prediction_by_prefix[60.0], bundle=bundle,
                lo=lo, hi=hi, thresholds=thresholds, decision=decision,
            ))

        # Importance weighting on the untouched calibration scores.  The final
        # test rows and outcomes are absent from density fitting and calibration.
        for scenario, bundle, prediction_by_prefix, reference in (
            ("clean", roles["test"], clean_pred, external["clean_reference"]),
            ("ood", external["ood_test"], ood_pred, external["ood_reference"]),
        ):
            widths: Dict[float, np.ndarray] = {}
            intervals: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            diagnostics = None
            for window in PREFIXES:
                cal_features = C.deployable_covariates(inputs_from_bundle(roles["calibration"], window))
                reference_features = C.deployable_covariates(inputs_from_bundle(reference, window))
                ratio = C.fit_density_ratio(cal_features, reference_features, seed=split_seed)
                iw = C.fit_iwcp(
                    cal_pred[window], roles["calibration"].true_I_900,
                    cal_features, ratio,
                    provenance=f"split={split_seed}/calibration+independent-{scenario}-reference",
                )
                test_features = C.deployable_covariates(inputs_from_bundle(bundle, window))
                lo, hi, _ = iw.interval(prediction_by_prefix[window], test_features)
                intervals[window] = (lo, hi)
                widths[window] = hi - lo
                if window == 60.0:
                    diagnostics = iw
            assert diagnostics is not None
            decision = decision_times(widths, thresholds.w_max)
            lo, hi = intervals[60.0]
            label = "WR-CP" if model_name.endswith("_wr") else "IW-CP"
            rows.append(_row(
                split_seed=split_seed, model_name=model_name,
                conformal_method=label, scenario=scenario,
                prediction=prediction_by_prefix[60.0], bundle=bundle,
                lo=lo, hi=hi, thresholds=thresholds, decision=decision,
                ess=diagnostics.ess,
                ess_fraction=diagnostics.ess_fraction,
                density_auc=diagnostics.density_ratio.cv_auc,
            ))

        if model_name in {"physics_only", "residual_only", "graybox"}:
            inferred = model.infer_cf0(inputs_from_bundle(roles["test"], 60.0))
            for row in rows:
                if row["split_seed"] == split_seed and row["model"] == model_name:
                    row["cf0_rmse_ng_ml"] = rmse(inferred, roles["test"].C_f0)
                    row["cf0_mae_ng_ml"] = mae(inferred, roles["test"].C_f0)
    return rows


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse_dn", "mae_dn", "coverage", "mean_width_dn", "abstention_rate",
        "full_diagnostic_accuracy", "selective_diagnostic_accuracy",
        "decision_time_mean_s", "decided_by_60_rate", "iw_ess", "iw_ess_fraction",
        "density_ratio_cv_auc", "cf0_rmse_ng_ml", "cf0_mae_ng_ml",
    ]
    present = [column for column in numeric if column in frame]
    grouped = frame.groupby(["model", "conformal", "scenario"], dropna=False)[present]
    mean = grouped.mean().reset_index()
    std = grouped.std().reset_index()
    mean["record_type"] = "aggregate_mean"
    std["record_type"] = "aggregate_sd"
    mean["split_seed"] = -1
    std["split_seed"] = -1
    return pd.concat([mean, std], ignore_index=True, sort=False)


def _frame_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("—" if np.isnan(value) else f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown(frame: pd.DataFrame, n_splits: int) -> str:
    aggregate = frame[frame.record_type == "aggregate_mean"].copy()
    primary = aggregate[
        (aggregate.scenario == "clean")
        & (((aggregate.conformal == "SCP") & aggregate.model.isin(["physics_only", "residual_only", "graybox", "tcn"]))
           | ((aggregate.conformal == "WR-CP") & aggregate.model.isin(["graybox_wr", "tcn_wr"])))
    ].copy()
    columns = [
        "model", "conformal", "rmse_dn", "mae_dn", "coverage", "mean_width_dn",
        "abstention_rate", "selective_diagnostic_accuracy", "decided_by_60_rate",
    ]
    primary = primary[columns].sort_values(["conformal", "model"])

    run = frame[frame.record_type == "run"]
    drops = []
    for keys, group in run.groupby(["split_seed", "model", "conformal"]):
        clean = group[group.scenario == "clean"]
        ood = group[group.scenario == "ood"]
        if len(clean) and len(ood):
            drops.append({
                "split_seed": keys[0], "model": keys[1], "conformal": keys[2],
                "ood_coverage_drop": float(clean.coverage.iloc[0] - ood.coverage.iloc[0]),
            })
    drop_frame = pd.DataFrame(drops)
    drop_summary = (
        drop_frame.groupby(["model", "conformal"]).ood_coverage_drop
        .agg(["mean", "std"]).reset_index()
        if len(drop_frame) else pd.DataFrame()
    )
    lines = [
        "# KinetiFlow-CP evaluation metrics",
        "",
        f"Results aggregate {n_splits} randomized, leakage-safe day/lot assignments. Values are means; see `metrics.csv` for every run and standard deviations.",
        "",
        "## Primary clean-test comparison",
        "",
        _frame_markdown(primary),
        "",
        "## Clean-to-OOD coverage drop",
        "",
        _frame_markdown(drop_summary) if len(drop_summary) else "No paired rows.",
        "",
        "## Validity note",
        "",
        "SCP's theorem requires exchangeability; held-out day/lot blocks with hierarchical effects do not establish it. These repeated assignments are empirical sensitivity evidence. WR-CP uses an independent unlabeled target-reference pool and never reads final test outcomes; its claim is limited to the multi-source/overlap assumptions, and ESS is reported for every run.",
        "",
    ]
    return "\n".join(lines)


def run_evaluation(n_splits: int, epochs: int, patience: int) -> pd.DataFrame:
    data = D.load()
    external = _external_sets()
    rows: List[Dict] = []
    started = time.time()
    for seed in range(n_splits):
        split_started = time.time()
        rows.extend(evaluate_split(data, external, seed, epochs, patience))
        elapsed = time.time() - split_started
        total = time.time() - started
        print(f"split {seed + 1:02d}/{n_splits}: {elapsed:.1f}s (total {total / 60:.1f} min)", flush=True)
    runs = pd.DataFrame(rows)
    aggregate = _aggregate(runs)
    frame = pd.concat([runs, aggregate], ignore_index=True, sort=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "metrics.md").write_text(_markdown(frame, n_splits))
    D.save_manifest(D.grouped_role_split(data, 0), 0, RESULTS_DIR / "split_manifest_seed0.json")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    if args.splits < 1:
        raise ValueError("--splits must be positive")
    frame = run_evaluation(args.splits, args.epochs, args.patience)
    primary = frame[
        (frame.record_type == "aggregate_mean")
        & (frame.scenario == "clean")
        & (frame.conformal == "SCP")
        & frame.model.isin(["physics_only", "residual_only", "graybox", "tcn"])
    ]
    print(primary[["model", "rmse_dn", "mae_dn", "coverage", "mean_width_dn"]].to_string(index=False))
    print(f"wrote {RESULTS_DIR / 'metrics.csv'}")
    print(f"wrote {RESULTS_DIR / 'metrics.md'}")


if __name__ == "__main__":
    main()
