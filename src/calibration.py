"""Optical calibration for the lateral-flow measurement model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import f as f_distribution


REQUIRED_COLUMNS = ("known_concentration", "measured_intensity")


@dataclass(frozen=True)
class FitDiagnostics:
    """Summary statistics for one calibration curve fit."""

    model: str
    alpha: float
    beta: float
    r2: float
    rmse: float
    ss_res: float
    aic: float
    n: int
    k: float | None = None

    def measurement_model_kwargs(self) -> dict[str, float | bool]:
        """Return kwargs accepted by mechanistic_ode.MeasurementModel."""

        kwargs: dict[str, float | bool] = {
            "alpha": self.alpha,
            "beta": self.beta,
            "sigmoidal": self.model == "sigmoidal",
        }
        if self.model == "sigmoidal":
            if self.k is None:
                raise ValueError("Sigmoidal fit is missing k")
            kwargs["k_sig"] = self.k
        return kwargs


@dataclass(frozen=True)
class NonlinearityTest:
    """Beer-Lambert linearity diagnostic for high-concentration curvature."""

    detected: bool
    f_statistic: float
    p_value: float
    quadratic_r2: float
    high_concentration_threshold: float
    high_residual_mean: float
    high_residual_z: float
    curvature: float
    saturation_like: bool
    reason: str


@dataclass(frozen=True)
class CalibrationResult:
    """Complete optical calibration result."""

    selected_model: str
    linear: FitDiagnostics
    nonlinearity_test: NonlinearityTest
    sigmoidal: FitDiagnostics | None

    def selected_fit(self) -> FitDiagnostics:
        """Return the fit selected for downstream MeasurementModel use."""

        if self.selected_model == "sigmoidal":
            if self.sigmoidal is None:
                raise ValueError("selected_model is sigmoidal but no sigmoidal fit exists")
            return self.sigmoidal
        return self.linear

    def measurement_model_kwargs(self) -> dict[str, float | bool]:
        """Return kwargs accepted by mechanistic_ode.MeasurementModel."""

        return self.selected_fit().measurement_model_kwargs()


def load_calibration_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a calibration CSV.

    The CSV must contain ``known_concentration`` and ``measured_intensity``.
    Rows with missing or non-finite values are dropped, and the remaining rows
    are sorted by concentration.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration CSV does not exist: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Calibration CSV is missing required columns: {missing}")

    data = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("known_concentration", kind="mergesort").reset_index(drop=True)

    if len(data) < 3:
        raise ValueError("At least 3 valid calibration points are required")
    if data["known_concentration"].nunique() < 2:
        raise ValueError("At least 2 distinct known concentrations are required")
    return data


def fit_linear_calibration(
    concentration: Sequence[float],
    intensity: Sequence[float],
) -> tuple[FitDiagnostics, np.ndarray, np.ndarray]:
    """Fit ``I_obs = alpha * C + beta`` by ordinary least squares."""

    x = _as_1d_float_array(concentration, "concentration")
    y = _as_1d_float_array(intensity, "intensity")
    _validate_xy(x, y, min_points=3)

    design = np.column_stack([x, np.ones_like(x)])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = alpha * x + beta
    residuals = y - predicted
    diagnostics = _diagnostics(
        model="linear",
        y=y,
        predicted=predicted,
        parameter_count=2,
        alpha=float(alpha),
        beta=float(beta),
    )
    return diagnostics, predicted, residuals


def test_beer_lambert_linearity(
    concentration: Sequence[float],
    intensity: Sequence[float],
    linear_predicted: Sequence[float],
    *,
    high_quantile: float = 0.75,
    p_threshold: float = 0.05,
    min_high_residual_z: float = 1.0,
) -> NonlinearityTest:
    """Test whether high-concentration points violate Beer-Lambert linearity.

    A quadratic lack-of-fit test checks curvature beyond the linear model. The
    result is marked as detected only when the curvature is statistically
    significant and the upper-concentration residuals are systematically shifted.
    ``saturation_like`` is true when curvature bends toward a plateau, allowing
    either positive or negative optical sign conventions.
    """

    x = _as_1d_float_array(concentration, "concentration")
    y = _as_1d_float_array(intensity, "intensity")
    y_hat = _as_1d_float_array(linear_predicted, "linear_predicted")
    _validate_xy(x, y, min_points=5)
    if y_hat.shape != y.shape:
        raise ValueError("linear_predicted must have the same length as intensity")

    residuals = y - y_hat
    x_center = float(np.mean(x))
    x_scale = float(np.std(x))
    if x_scale <= 0:
        raise ValueError("Concentration values must vary")
    xs = (x - x_center) / x_scale

    quadratic_design = np.column_stack([xs, xs**2, np.ones_like(xs)])
    q_slope, q_curvature, q_beta = np.linalg.lstsq(quadratic_design, y, rcond=None)[0]
    quadratic_predicted = q_slope * xs + q_curvature * xs**2 + q_beta

    ss_linear = _ss_res(y, y_hat)
    ss_quadratic = _ss_res(y, quadratic_predicted)
    df_den = len(y) - 3
    if df_den <= 0 or ss_quadratic <= 0 or ss_linear <= ss_quadratic:
        f_stat = 0.0
        p_value = 1.0
    else:
        f_stat = ((ss_linear - ss_quadratic) / 1.0) / (ss_quadratic / df_den)
        p_value = float(f_distribution.sf(f_stat, 1, df_den))

    high_threshold = float(np.quantile(x, high_quantile))
    high_mask = x >= high_threshold
    high_residual_mean = float(np.mean(residuals[high_mask]))
    rmse = float(np.sqrt(ss_linear / max(len(y) - 2, 1)))
    high_se = rmse / np.sqrt(max(int(np.sum(high_mask)), 1))
    high_residual_z = abs(high_residual_mean) / high_se if high_se > 0 else 0.0

    linear_slope = float(np.polyfit(xs, y, deg=1)[0])
    saturation_like = bool(linear_slope * float(q_curvature) < 0)
    detected = bool(
        p_value < p_threshold
        and high_residual_z >= min_high_residual_z
        and saturation_like
    )
    if detected:
        reason = (
            "significant saturation-like curvature with shifted high-concentration "
            "linear residuals"
        )
    elif p_value < p_threshold and high_residual_z >= min_high_residual_z:
        reason = "significant curvature, but it is not plateau-like"
    elif p_value < p_threshold:
        reason = "significant curvature, but high-concentration residual shift is weak"
    else:
        reason = "no significant high-concentration curvature"

    return NonlinearityTest(
        detected=detected,
        f_statistic=float(f_stat),
        p_value=float(p_value),
        quadratic_r2=_r2_score(y, quadratic_predicted),
        high_concentration_threshold=high_threshold,
        high_residual_mean=high_residual_mean,
        high_residual_z=float(high_residual_z),
        curvature=float(q_curvature),
        saturation_like=saturation_like,
        reason=reason,
    )


def fit_sigmoidal_calibration(
    concentration: Sequence[float],
    intensity: Sequence[float],
) -> tuple[FitDiagnostics, np.ndarray, np.ndarray]:
    """Fit ``I_obs = alpha * tanh(k * C) + beta`` for saturation."""

    x = _as_1d_float_array(concentration, "concentration")
    y = _as_1d_float_array(intensity, "intensity")
    _validate_xy(x, y, min_points=4)

    x_max = float(np.max(np.abs(x)))
    if x_max <= 0:
        raise ValueError("At least one concentration must be non-zero")

    beta0 = float(y[np.argmin(np.abs(x))]) if np.any(np.isclose(x, 0.0)) else float(y[0])
    response_span = float(y[-1] - beta0)
    if abs(response_span) < 1e-8:
        response_span = float(np.max(y) - np.min(y))
    alpha0 = response_span if abs(response_span) >= 1e-8 else 1.0
    slope0 = float(np.polyfit(x, y, deg=1)[0])
    k0 = abs(slope0 / alpha0) if abs(alpha0) > 1e-8 else 1.0 / x_max
    k0 = float(np.clip(k0, 1e-8, 100.0 / x_max))

    lower = [-np.inf, 1e-12, -np.inf]
    upper = [np.inf, np.inf, np.inf]
    popt, _ = curve_fit(
        _sigmoidal_model,
        x,
        y,
        p0=[alpha0, k0, beta0],
        bounds=(lower, upper),
        maxfev=20000,
    )
    alpha, k, beta = (float(v) for v in popt)
    predicted = _sigmoidal_model(x, alpha, k, beta)
    residuals = y - predicted
    diagnostics = _diagnostics(
        model="sigmoidal",
        y=y,
        predicted=predicted,
        parameter_count=3,
        alpha=alpha,
        beta=beta,
        k=k,
    )
    return diagnostics, predicted, residuals


def calibrate_optics(
    csv_path: str | Path,
    *,
    force_sigmoidal: bool = False,
    high_quantile: float = 0.75,
    p_threshold: float = 0.05,
    min_high_residual_z: float = 1.0,
) -> tuple[CalibrationResult, pd.DataFrame]:
    """Fit optical calibration and return the selected result plus residual table."""

    data = load_calibration_csv(csv_path)
    concentration = data["known_concentration"].to_numpy(dtype=float)
    intensity = data["measured_intensity"].to_numpy(dtype=float)

    linear, linear_predicted, linear_residuals = fit_linear_calibration(
        concentration,
        intensity,
    )
    nonlinearity = test_beer_lambert_linearity(
        concentration,
        intensity,
        linear_predicted,
        high_quantile=high_quantile,
        p_threshold=p_threshold,
        min_high_residual_z=min_high_residual_z,
    )

    residual_table = data.copy()
    residual_table["linear_prediction"] = linear_predicted
    residual_table["linear_residual"] = linear_residuals

    sigmoidal: FitDiagnostics | None = None
    selected_model = "linear"
    if nonlinearity.detected or force_sigmoidal:
        sigmoidal, sig_predicted, sig_residuals = fit_sigmoidal_calibration(
            concentration,
            intensity,
        )
        residual_table["sigmoidal_prediction"] = sig_predicted
        residual_table["sigmoidal_residual"] = sig_residuals

        # Prefer the extra parameter only when it gives a meaningful AIC gain.
        selected_model = "sigmoidal" if sigmoidal.aic + 2.0 < linear.aic else "linear"

    result = CalibrationResult(
        selected_model=selected_model,
        linear=linear,
        nonlinearity_test=nonlinearity,
        sigmoidal=sigmoidal,
    )
    return result, residual_table


def save_calibration_outputs(
    result: CalibrationResult,
    residual_table: pd.DataFrame,
    *,
    output_json: str | Path,
    residuals_csv: str | Path,
) -> None:
    """Save calibration parameters and residual diagnostics."""

    output_json = Path(output_json)
    residuals_csv = Path(residuals_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    residuals_csv.parent.mkdir(parents=True, exist_ok=True)

    report = calibration_report_dict(result)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    residual_table.to_csv(residuals_csv, index=False)


def calibration_report_dict(result: CalibrationResult) -> dict[str, Any]:
    """Convert a calibration result into a JSON-serializable report."""

    report: dict[str, Any] = {
        "selected_model": result.selected_model,
        "measurement_model_kwargs": result.measurement_model_kwargs(),
        "linear": asdict(result.linear),
        "nonlinearity_test": asdict(result.nonlinearity_test),
        "sigmoidal": asdict(result.sigmoidal) if result.sigmoidal is not None else None,
        "usage": (
            "kwargs = json.load(open(path))['measurement_model_kwargs']; "
            "meas = MeasurementModel(**kwargs)"
        ),
    }
    if result.sigmoidal is not None:
        report["fit_comparison"] = {
            "delta_aic_sigmoidal_minus_linear": result.sigmoidal.aic - result.linear.aic,
            "delta_r2_sigmoidal_minus_linear": result.sigmoidal.r2 - result.linear.r2,
        }
    return report


def plot_calibration_diagnostics(
    result: CalibrationResult,
    residual_table: pd.DataFrame,
    output_png: str | Path,
) -> None:
    """Save a diagnostic plot of calibration fits and residuals."""

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = residual_table["known_concentration"].to_numpy(dtype=float)
    y = residual_table["measured_intensity"].to_numpy(dtype=float)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)

    fig, (ax_fit, ax_resid) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    ax_fit.scatter(x, y, s=36, color="black", label="measurements", zorder=3)
    ax_fit.plot(
        x_grid,
        result.linear.alpha * x_grid + result.linear.beta,
        color="C0",
        lw=2,
        label=f"linear R2={result.linear.r2:.4f}",
    )
    ax_resid.axhline(0.0, color="0.25", lw=1)
    ax_resid.scatter(
        x,
        residual_table["linear_residual"],
        color="C0",
        s=30,
        label="linear residual",
    )

    if result.sigmoidal is not None:
        k = result.sigmoidal.k
        if k is None:
            raise ValueError("Sigmoidal result is missing k")
        ax_fit.plot(
            x_grid,
            _sigmoidal_model(x_grid, result.sigmoidal.alpha, k, result.sigmoidal.beta),
            color="C3",
            lw=2,
            label=f"sigmoidal R2={result.sigmoidal.r2:.4f}",
        )
        ax_resid.scatter(
            x,
            residual_table["sigmoidal_residual"],
            color="C3",
            s=30,
            label="sigmoidal residual",
        )

    threshold = result.nonlinearity_test.high_concentration_threshold
    ax_fit.axvline(threshold, color="0.5", ls="--", lw=1, alpha=0.8)
    ax_resid.axvline(threshold, color="0.5", ls="--", lw=1, alpha=0.8)

    ax_fit.set_ylabel("Measured intensity (DN)")
    ax_fit.set_title(f"Optical calibration: selected {result.selected_model}")
    ax_fit.legend()
    ax_fit.grid(alpha=0.25)

    ax_resid.set_xlabel("Known concentration")
    ax_resid.set_ylabel("Residual (DN)")
    ax_resid.legend()
    ax_resid.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_synthetic_calibration_csv(csv_path: str | Path, *, seed: int = 11) -> Path:
    """Write a synthetic calibration CSV that exercises high-dose saturation."""

    rng = np.random.default_rng(seed)
    concentration = np.concatenate(
        [
            np.linspace(0.0, 8.0, 9),
            np.linspace(10.0, 45.0, 12),
        ]
    )
    alpha_true = 180.0
    beta_true = 24.0
    k_true = 0.105
    intensity = _sigmoidal_model(concentration, alpha_true, k_true, beta_true)
    intensity += rng.normal(0.0, 1.8, size=concentration.shape)

    df = pd.DataFrame(
        {
            "known_concentration": concentration,
            "measured_intensity": intensity,
        }
    )
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _diagnostics(
    *,
    model: str,
    y: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
    alpha: float,
    beta: float,
    k: float | None = None,
) -> FitDiagnostics:
    ss_res = _ss_res(y, predicted)
    rmse = float(np.sqrt(ss_res / max(len(y) - parameter_count, 1)))
    return FitDiagnostics(
        model=model,
        alpha=float(alpha),
        beta=float(beta),
        k=None if k is None else float(k),
        r2=_r2_score(y, predicted),
        rmse=rmse,
        ss_res=ss_res,
        aic=_aic(ss_res, len(y), parameter_count),
        n=int(len(y)),
    )


def _sigmoidal_model(
    concentration: np.ndarray,
    alpha: float,
    k: float,
    beta: float,
) -> np.ndarray:
    return alpha * np.tanh(k * concentration) + beta


def _ss_res(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((y - predicted) ** 2))


def _r2_score(y: np.ndarray, predicted: np.ndarray) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0 if np.allclose(y, predicted) else 0.0
    return 1.0 - (_ss_res(y, predicted) / ss_tot)


def _aic(ss_res: float, n: int, parameter_count: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    variance = max(ss_res / n, np.finfo(float).tiny)
    return float(n * np.log(variance) + 2 * parameter_count)


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_xy(x: np.ndarray, y: np.ndarray, *, min_points: int) -> None:
    if x.shape != y.shape:
        raise ValueError("concentration and intensity must have the same length")
    if len(x) < min_points:
        raise ValueError(f"At least {min_points} points are required")
    if np.ptp(x) <= 0:
        raise ValueError("Concentration values must vary")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit optical calibration for MeasurementModel.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Calibration CSV with known_concentration, measured_intensity.",
    )
    parser.add_argument(
        "--synthetic-csv",
        default="/tmp/kinetiflow_synthetic_calibration.csv",
        help="Synthetic CSV path used when no input CSV is provided.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/kinetiflow_calibration_fit.json",
        help="JSON output with MeasurementModel kwargs.",
    )
    parser.add_argument(
        "--residuals-csv",
        default="/tmp/kinetiflow_calibration_residuals.csv",
        help="Residual diagnostics CSV output path.",
    )
    parser.add_argument(
        "--plot",
        default="/tmp/kinetiflow_calibration_diagnostic.png",
        help="Diagnostic plot PNG output path.",
    )
    parser.add_argument(
        "--force-sigmoidal",
        action="store_true",
        help="Fit the sigmoidal model even if nonlinearity is not detected.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the diagnostic plot.",
    )
    return parser


def main() -> None:
    """Run calibration from the command line."""

    args = _build_arg_parser().parse_args()
    if args.csv is None:
        csv_path = write_synthetic_calibration_csv(args.synthetic_csv)
        print(f"Wrote synthetic calibration CSV: {csv_path}")
    else:
        csv_path = Path(args.csv)

    result, residual_table = calibrate_optics(
        csv_path,
        force_sigmoidal=args.force_sigmoidal,
    )
    save_calibration_outputs(
        result,
        residual_table,
        output_json=args.output_json,
        residuals_csv=args.residuals_csv,
    )
    if not args.no_plot:
        plot_calibration_diagnostics(result, residual_table, args.plot)

    selected = result.selected_fit()
    print(f"Selected model: {result.selected_model}")
    print(f"alpha={selected.alpha:.8g}, beta={selected.beta:.8g}")
    if selected.k is not None:
        print(f"k={selected.k:.8g}  (MeasurementModel kwarg: k_sig)")
    print(f"R2={selected.r2:.6f}, RMSE={selected.rmse:.6f}, AIC={selected.aic:.6f}")
    print(
        "Nonlinearity test: "
        f"detected={result.nonlinearity_test.detected}, "
        f"p={result.nonlinearity_test.p_value:.6g}, "
        f"reason={result.nonlinearity_test.reason}"
    )
    print(f"Wrote calibration JSON: {args.output_json}")
    print(f"Wrote residuals CSV: {args.residuals_csv}")
    if not args.no_plot:
        print(f"Wrote diagnostic plot: {args.plot}")


if __name__ == "__main__":
    main()
