"""Convert lateral-flow strip videos or frame folders into intensity traces."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROI = tuple[int, int, int, int]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
RAW_ARRAY_EXTENSIONS = {".npy"}


@dataclass(frozen=True)
class FrameSample:
    """One OpenCV frame with a millisecond timestamp."""

    t_ms: float
    frame: np.ndarray
    source: str | None = None


def load_frames(source: str | Path, folder_fps: float = 30.0) -> list[FrameSample]:
    """Load a video, image, or folder of frames with millisecond timestamps.

    Video timestamps come from OpenCV's ``CAP_PROP_POS_MSEC`` with an FPS-based
    fallback. Frame-folder timestamps are parsed from filenames containing
    ``...ms`` or ``...s``; otherwise they are assigned from ``folder_fps``.
    """

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if folder_fps <= 0:
        raise ValueError("folder_fps must be positive")

    if path.is_dir():
        return _load_frame_folder(path, folder_fps)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        return [FrameSample(t_ms=0.0, frame=frame, source=str(path))]
    return _load_video(path)


def pick_roi_interactive(frame: np.ndarray, window_name: str = "Select ROI") -> ROI:
    """Let the user draw a rectangular ROI with OpenCV's interactive selector."""

    x, y, w, h = cv2.selectROI(
        window_name,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(window_name)
    roi = (int(x), int(y), int(w), int(h))
    _validate_roi(frame, roi)
    return roi


def detect_munsell_n5_patch_roi(
    frame: np.ndarray,
    *,
    target_dn: float = 128.0,
    min_area_px: int = 100,
    max_area_fraction: float = 0.20,
) -> ROI:
    """Detect a neutral, mid-gray Munsell N5-like reference patch ROI.

    The detector favors low-saturation, medium-brightness rectangular regions
    whose grayscale mean is near ``target_dn``. Pass ``grey_roi`` explicitly to
    the pipeline when the patch geometry is known.
    """

    _require_nonempty_frame(frame)
    detection_frame = _to_uint8_bgr(frame)
    hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY).astype(float)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (
        (saturation <= 45)
        & (value >= max(0, int(target_dn) - 50))
        & (value <= min(255, int(target_dn) + 50))
    ).astype(np.uint8)
    mask *= 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_roi: ROI | None = None
    best_score = -np.inf
    max_area_px = float(frame.shape[0] * frame.shape[1]) * max_area_fraction
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px or area > max_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        rect_area = float(w * h)
        fill = area / rect_area
        if fill < 0.45:
            continue
        roi_gray = gray[y : y + h, x : x + w]
        roi_sat = saturation[y : y + h, x : x + w]
        mean_dn = float(np.mean(roi_gray))
        mean_sat = float(np.mean(roi_sat))
        aspect_penalty = abs(np.log(max(w / h, h / w)))
        score = (0.25 * np.sqrt(area)) + (100.0 * fill)
        score -= 2.0 * abs(mean_dn - target_dn)
        score -= 0.45 * mean_sat + 12.0 * aspect_penalty
        if score > best_score:
            best_score = score
            best_roi = (int(x), int(y), int(w), int(h))

    if best_roi is None:
        raise ValueError(
            "Could not detect a Munsell N5 grey patch. Pass grey_roi=(x, y, w, h)."
        )
    _validate_roi(frame, best_roi)
    return best_roi


def extract_intensity_timeseries(
    samples: Sequence[FrameSample],
    test_roi: ROI,
    *,
    grey_roi: ROI | None = None,
    max_grey_drift_dn: float = 3.0,
    smooth_window: int = 15,
    smooth_polyorder: int = 3,
) -> pd.DataFrame:
    """Extract illumination-normalized raw and Savitzky-Golay-smoothed intensity.

    ``test_roi`` and ``grey_roi`` use OpenCV coordinates ``(x, y, width, height)``.
    If ``grey_roi`` is omitted, the first frame is searched for a Munsell N5-like
    patch. Any frame whose grey-patch mean differs by more than
    ``max_grey_drift_dn`` from the first frame is rejected.
    """

    if not samples:
        raise ValueError("No frames were provided")
    first_frame = samples[0].frame
    _validate_roi(first_frame, test_roi)
    if grey_roi is None:
        grey_roi = detect_munsell_n5_patch_roi(first_frame)
    _validate_roi(first_frame, grey_roi)

    reference_grey = mean_roi_intensity(first_frame, grey_roi)
    if reference_grey <= 0:
        raise ValueError("Grey reference intensity must be positive")

    t_seconds: list[float] = []
    intensities: list[float] = []
    for sample in samples:
        _validate_roi(sample.frame, test_roi)
        _validate_roi(sample.frame, grey_roi)
        grey_mean = mean_roi_intensity(sample.frame, grey_roi)
        if abs(grey_mean - reference_grey) > max_grey_drift_dn:
            continue
        test_mean = mean_roi_intensity(sample.frame, test_roi)
        normalized_test_mean = test_mean * (reference_grey / grey_mean)
        t_seconds.append(sample.t_ms / 1000.0)
        intensities.append(float(normalized_test_mean))

    if not intensities:
        raise ValueError("All frames were rejected by grey-patch drift screening")

    raw = np.asarray(intensities, dtype=float)
    smoothed = smooth_intensity(
        raw,
        window_length=smooth_window,
        polyorder=smooth_polyorder,
    )
    return pd.DataFrame(
        {
            "t_seconds": np.asarray(t_seconds, dtype=float),
            "I_raw": raw,
            "I_smoothed": smoothed,
        }
    )


def process_lateral_flow_source(
    source: str | Path,
    test_roi: ROI,
    output_csv: str | Path,
    *,
    grey_roi: ROI | None = None,
    folder_fps: float = 30.0,
    max_grey_drift_dn: float = 3.0,
    smooth_window: int = 15,
    smooth_polyorder: int = 3,
) -> pd.DataFrame:
    """Load a source, extract ``I_obs(t)``, save CSV, and return the DataFrame."""

    samples = load_frames(source, folder_fps=folder_fps)
    df = extract_intensity_timeseries(
        samples,
        test_roi,
        grey_roi=grey_roi,
        max_grey_drift_dn=max_grey_drift_dn,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
    )
    save_timeseries_csv(df, output_csv)
    return df


def mean_roi_intensity(frame: np.ndarray, roi: ROI) -> float:
    """Return the mean grayscale digital number inside an ROI."""

    _validate_roi(frame, roi)
    x, y, w, h = roi
    roi_gray = _to_gray(frame)[y : y + h, x : x + w]
    return float(np.mean(roi_gray))


def smooth_intensity(
    intensities: Sequence[float],
    *,
    window_length: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth an intensity trace with a Savitzky-Golay filter.

    The default ``window_length=15`` and ``polyorder=3`` preserve local curvature
    and inflection points better than a moving average. For short traces, the
    window is reduced to the largest valid odd length.
    """

    values = np.asarray(intensities, dtype=float)
    if values.ndim != 1:
        raise ValueError("intensities must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be odd")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")

    effective_window = min(window_length, values.size)
    if effective_window % 2 == 0:
        effective_window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if effective_window < min_window:
        return values.copy()
    return savgol_filter(
        values,
        window_length=effective_window,
        polyorder=polyorder,
        mode="interp",
    )


def save_timeseries_csv(df: pd.DataFrame, output_csv: str | Path) -> None:
    """Save the intensity DataFrame as CSV."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_timeseries(df: pd.DataFrame) -> None:
    """Plot raw and smoothed intensity traces."""

    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["t_seconds"], df["I_raw"], "o", ms=3, alpha=0.55, label="raw")
    ax.plot(df["t_seconds"], df["I_smoothed"], "-", lw=2, label="smoothed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity (DN)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def _load_frame_folder(path: Path, folder_fps: float) -> list[FrameSample]:
    frame_paths = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | RAW_ARRAY_EXTENSIONS
    )
    if not frame_paths:
        raise ValueError(f"No image or raw .npy frames found in folder: {path}")

    samples: list[FrameSample] = []
    for idx, frame_path in enumerate(frame_paths):
        if frame_path.suffix.lower() in RAW_ARRAY_EXTENSIONS:
            frame = np.load(frame_path, allow_pickle=False)
            if frame.ndim not in (2, 3) or not np.issubdtype(frame.dtype, np.number):
                raise ValueError(f"Unsupported raw frame array {frame_path}: {frame.shape} {frame.dtype}")
        else:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
        t_ms = _timestamp_ms_from_name(frame_path)
        if t_ms is None:
            t_ms = 1000.0 * idx / folder_fps
        samples.append(FrameSample(t_ms=float(t_ms), frame=frame, source=str(frame_path)))
    if not samples:
        raise ValueError(f"OpenCV could not read any frames in folder: {path}")
    return samples


def _load_video(path: Path) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    samples: list[FrameSample] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if (not np.isfinite(t_ms) or t_ms <= 0.0) and idx > 0 and fps > 0.0:
            t_ms = 1000.0 * idx / fps
        samples.append(FrameSample(t_ms=t_ms, frame=frame, source=str(path)))
        idx += 1
    cap.release()

    if not samples:
        raise ValueError(f"No frames were read from video: {path}")
    return samples


def _timestamp_ms_from_name(path: Path) -> float | None:
    stem = path.stem.lower()
    ms_match = re.search(r"(\d+(?:\.\d+)?)\s*ms\b", stem)
    if ms_match:
        return float(ms_match.group(1))
    s_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:secs|sec|s)\b", stem)
    if s_match:
        return 1000.0 * float(s_match.group(1))
    return None


def _parse_roi(values: Sequence[int] | None) -> ROI | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI must contain four values: x y width height")
    x, y, w, h = (int(v) for v in values)
    return (x, y, w, h)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    _require_nonempty_frame(frame)
    if frame.ndim == 2:
        return frame.astype(float, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY).astype(float, copy=False)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    _require_nonempty_frame(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _to_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    """Scale uint10/12/16 raw arrays to uint8 for ROI detection only.

    Quantitative ROI intensities continue to use the original sensor values.
    """
    bgr = _as_bgr(frame)
    if bgr.dtype == np.uint8:
        return bgr
    values = bgr.astype(np.float32)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Raw frame has no usable dynamic range")
    scaled = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0)
    return scaled.astype(np.uint8)


def _validate_roi(frame: np.ndarray, roi: ROI) -> None:
    _require_nonempty_frame(frame)
    x, y, w, h = roi
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI {roi}; expected x>=0, y>=0, w>0, h>0")
    if x + w > width or y + h > height:
        raise ValueError(f"ROI {roi} exceeds frame bounds {(width, height)}")


def _require_nonempty_frame(frame: np.ndarray) -> None:
    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty")


def _make_synthetic_samples(num_frames: int = 60) -> tuple[list[FrameSample], ROI, ROI]:
    rng = np.random.default_rng(7)
    height, width = 180, 320
    test_roi: ROI = (118, 82, 86, 14)
    grey_roi: ROI = (24, 24, 42, 30)
    samples: list[FrameSample] = []

    for idx in range(num_frames):
        frame = np.full((height, width, 3), 232, dtype=np.uint8)
        t_s = idx / 2.0
        grey_dn = 128 + 1.4 * np.sin(idx / 11.0)
        line_dn = 218 - 74 * (1.0 - np.exp(-t_s / 10.0))
        line_dn += 3.0 * np.sin(t_s / 3.3)

        gx, gy, gw, gh = grey_roi
        tx, ty, tw, th = test_roi
        frame[gy : gy + gh, gx : gx + gw] = np.clip(grey_dn, 0, 255)
        frame[ty : ty + th, tx : tx + tw] = np.clip(line_dn, 0, 255)

        noise = rng.normal(0.0, 1.2, size=frame.shape)
        noisy = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
        samples.append(FrameSample(t_ms=1000.0 * t_s, frame=noisy, source="synthetic"))

    return samples, test_roi, grey_roi


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a lateral-flow test-line intensity time-series.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Video, single image, or folder of frames. If omitted/missing, runs synthetic self-test.",
    )
    parser.add_argument(
        "--test-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Test-line ROI in OpenCV coordinates.",
    )
    parser.add_argument(
        "--grey-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Grey reference patch ROI. If omitted, the first frame is searched.",
    )
    parser.add_argument(
        "--pick-test-roi",
        action="store_true",
        help="Interactively draw the test-line ROI on the first frame.",
    )
    parser.add_argument(
        "--pick-grey-roi",
        action="store_true",
        help="Interactively draw the grey reference ROI on the first frame.",
    )
    parser.add_argument(
        "--output-csv",
        default="data_pipeline_output.csv",
        help="CSV output path.",
    )
    parser.add_argument("--folder-fps", type=float, default=30.0)
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot.")
    return parser


def main() -> None:
    """Run the pipeline from the command line."""

    args = _build_arg_parser().parse_args()
    source = Path(args.source) if args.source else None
    output_csv = Path(args.output_csv)

    if source is None or not source.exists():
        samples, test_roi, grey_roi = _make_synthetic_samples()
        df = extract_intensity_timeseries(samples, test_roi, grey_roi=grey_roi)
        save_timeseries_csv(df, output_csv)
        print(
            f"Synthetic self-test wrote {len(df)} rows to {output_csv} "
            f"using test_roi={test_roi}, grey_roi={grey_roi}"
        )
    else:
        samples = load_frames(source, folder_fps=args.folder_fps)
        first_frame = samples[0].frame
        test_roi = _parse_roi(args.test_roi)
        grey_roi = _parse_roi(args.grey_roi)
        if args.pick_test_roi:
            test_roi = pick_roi_interactive(first_frame, "Select test-line ROI")
        if test_roi is None:
            raise SystemExit(
                "Provide --test-roi X Y W H or use --pick-test-roi for a real source."
            )
        if args.pick_grey_roi:
            grey_roi = pick_roi_interactive(first_frame, "Select grey-reference ROI")
        df = extract_intensity_timeseries(samples, test_roi, grey_roi=grey_roi)
        save_timeseries_csv(df, output_csv)
        print(f"Wrote {len(df)} rows to {output_csv}")

    if not args.no_plot:
        plot_timeseries(df)


if __name__ == "__main__":
    main()
