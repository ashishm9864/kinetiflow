#!/usr/bin/env python3
"""Headless Raspberry Pi HQ Camera capture for lateral-flow strip videos.

This script is intended to run on the Raspberry Pi, not on the development
machine. It uses Picamera2/libcamera with locked optical settings and saves raw
Bayer frames as ``.npy`` arrays plus timestamp/experiment metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# Locked acquisition settings for the IMX477 HQ Camera.
# Picamera2/libcamera does not expose a portable "ISO" knob; analogue gain 1.0
# is the base-gain setting conventionally treated as ISO 100 on the HQ Camera.
LOCKED_ISO_EQUIVALENT = 100
LOCKED_ANALOGUE_GAIN = 1.0
LOCKED_SHUTTER_US = 2_000  # 1/500 s
LOCKED_AWB_ENABLE = False
LOCKED_COLOUR_GAINS = (1.0, 1.0)

# Full-resolution IMX477 raw frame size. Override only if the Pi reports a
# different sensor mode or you intentionally want smaller raw frames.
DEFAULT_RAW_WIDTH = 4056
DEFAULT_RAW_HEIGHT = 3040


@dataclass(frozen=True)
class CaptureSettings:
    """Camera and recording settings stored in metadata JSON."""

    raw_width: int
    raw_height: int
    fps: float
    duration_s: float
    warmup_s: float
    shutter_us: int
    iso_equivalent: int
    analogue_gain: float
    awb_enable: bool
    colour_gains: tuple[float, float]
    frame_duration_us: int


@dataclass(frozen=True)
class ExperimentArgs:
    """User-supplied experiment metadata."""

    strip_lot: str
    strip_date: str
    temperature_c: float
    humidity_rh: float


@dataclass(frozen=True)
class FrameRecord:
    """One saved raw frame and its timestamps."""

    frame_index: int
    t_ms: float
    sensor_timestamp_ns: int
    wall_time_iso: str
    raw_path: str
    exposure_time_us: int | None
    analogue_gain: float | None


def import_picamera2() -> Any:
    """Import Picamera2 with a clear off-Pi error."""

    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise SystemExit(
            "Picamera2 is not installed. Run this script on the Raspberry Pi "
            "with the Raspberry Pi camera stack enabled, for example: "
            "sudo apt install python3-picamera2"
        ) from exc
    return Picamera2


def configure_camera(picam2: Any, settings: CaptureSettings) -> dict[str, Any]:
    """Configure the HQ Camera for headless raw capture with locked controls."""

    raw_stream = {
        "size": (settings.raw_width, settings.raw_height),
    }
    # A small main stream keeps Picamera2 happy without showing any preview. The
    # analysis data are read from the raw stream below, not from this RGB stream.
    main_stream = {
        "size": (640, 480),
        "format": "RGB888",
    }

    config = picam2.create_video_configuration(
        main=main_stream,
        raw=raw_stream,
        controls={
            "FrameDurationLimits": (
                settings.frame_duration_us,
                settings.frame_duration_us,
            ),
        },
        buffer_count=4,
        queue=False,
    )
    picam2.configure(config)

    # Disable auto-exposure and AWB so intensity changes come from the strip,
    # not camera adaptation. These control names are libcamera/Picamera2 names.
    picam2.set_controls(
        {
            "AeEnable": False,
            "ExposureTime": settings.shutter_us,
            "AnalogueGain": settings.analogue_gain,
            "AwbEnable": settings.awb_enable,
            "ColourGains": settings.colour_gains,
            "FrameDurationLimits": (
                settings.frame_duration_us,
                settings.frame_duration_us,
            ),
        }
    )
    return _json_safe(picam2.camera_configuration())


def record_frames(
    picam2: Any,
    run_dir: Path,
    settings: CaptureSettings,
) -> tuple[list[FrameRecord], bool]:
    """Capture raw frames until ``duration_s`` elapses."""

    frames_dir = run_dir / "frames_raw"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[FrameRecord] = []
    interrupted = False

    first_sensor_timestamp_ns: int | None = None
    frame_interval_s = 1.0 / settings.fps
    next_capture_monotonic = time.monotonic()
    deadline = next_capture_monotonic + settings.duration_s

    frame_index = 0
    try:
        while time.monotonic() < deadline:
            request = picam2.capture_request()
            try:
                metadata = request.get_metadata()
                sensor_timestamp_ns = int(
                    metadata.get("SensorTimestamp", time.monotonic_ns())
                )
                if first_sensor_timestamp_ns is None:
                    first_sensor_timestamp_ns = sensor_timestamp_ns
                t_ms = (sensor_timestamp_ns - first_sensor_timestamp_ns) / 1_000_000.0

                # This is the raw Bayer stream, not a demosaiced RGB frame.
                # Saving as .npy preserves the sensor values and dtype without
                # image-codec surprises; downstream conversion can decide how to
                # demosaic.
                raw_frame = request.make_array("raw")
                raw_name = f"frame_{frame_index:06d}_{int(round(t_ms)):010d}ms.npy"
                raw_path = frames_dir / raw_name
                np.save(raw_path, raw_frame)

                frame_records.append(
                    FrameRecord(
                        frame_index=frame_index,
                        t_ms=float(t_ms),
                        sensor_timestamp_ns=sensor_timestamp_ns,
                        wall_time_iso=datetime.now().astimezone().isoformat(
                            timespec="milliseconds"
                        ),
                        raw_path=str(raw_path.relative_to(run_dir)),
                        exposure_time_us=_optional_int(metadata.get("ExposureTime")),
                        analogue_gain=_optional_float(metadata.get("AnalogueGain")),
                    )
                )
            finally:
                request.release()

            frame_index += 1
            next_capture_monotonic += frame_interval_s
            sleep_s = next_capture_monotonic - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        interrupted = True

    return frame_records, interrupted


def write_frames_csv(run_dir: Path, frame_records: list[FrameRecord]) -> Path:
    """Write per-frame timestamps to ``frames.csv``."""

    path = run_dir / "frames.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "t_ms",
                "sensor_timestamp_ns",
                "wall_time_iso",
                "raw_path",
                "exposure_time_us",
                "analogue_gain",
            ],
        )
        writer.writeheader()
        for record in frame_records:
            writer.writerow(asdict(record))
    return path


def write_metadata_json(
    run_dir: Path,
    experiment: ExperimentArgs,
    settings: CaptureSettings,
    camera_config: dict[str, Any],
    frame_records: list[FrameRecord],
    *,
    interrupted: bool,
) -> Path:
    """Write run-level metadata and frame log to ``metadata.json``."""

    metadata = {
        "script": "capture/record_strip.py",
        "camera": {
            "model": "Raspberry Pi HQ Camera / Sony IMX477",
            "stack": "Picamera2/libcamera",
            "headless": True,
            "configured_streams": camera_config,
        },
        "experiment": asdict(experiment),
        "capture_settings": asdict(settings),
        "recording": {
            "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "interrupted": interrupted,
            "frame_count": len(frame_records),
            "frames_csv": "frames.csv",
            "raw_frames_dir": "frames_raw",
        },
        "frames": [asdict(record) for record in frame_records],
        "notes": [
            "Raw frames are saved as NumPy arrays from the Picamera2 raw stream.",
            "Exposure, analogue gain, and white balance are locked for optical calibration.",
        ],
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(_json_safe(metadata), indent=2) + "\n")
    return path


def build_run_dir(base_dir: Path, experiment: ExperimentArgs) -> Path:
    """Create a unique output directory for one strip recording."""

    lot = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment.strip_lot).strip("_")
    lot = lot or "unknown-lot"
    stamp = datetime.now().strftime("%H%M%S")
    run_dir = base_dir / f"{experiment.strip_date}_{lot}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_capture_settings(args: argparse.Namespace) -> CaptureSettings:
    """Translate CLI arguments into locked capture settings."""

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if args.warmup_s < 0:
        raise SystemExit("--warmup-s must be non-negative")
    if args.raw_width <= 0 or args.raw_height <= 0:
        raise SystemExit("--raw-width and --raw-height must be positive")

    requested_frame_duration_us = int(round(1_000_000.0 / args.fps))
    frame_duration_us = max(requested_frame_duration_us, LOCKED_SHUTTER_US + 1_000)

    return CaptureSettings(
        raw_width=int(args.raw_width),
        raw_height=int(args.raw_height),
        fps=float(args.fps),
        duration_s=float(args.duration_s),
        warmup_s=float(args.warmup_s),
        shutter_us=LOCKED_SHUTTER_US,
        iso_equivalent=LOCKED_ISO_EQUIVALENT,
        analogue_gain=LOCKED_ANALOGUE_GAIN,
        awb_enable=LOCKED_AWB_ENABLE,
        colour_gains=LOCKED_COLOUR_GAINS,
        frame_duration_us=frame_duration_us,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a headless Pi capture run."""

    parser = argparse.ArgumentParser(
        description="Record raw IMX477 lateral-flow strip frames with locked settings.",
    )
    parser.add_argument("--strip-lot", required=True, help="Strip lot identifier.")
    parser.add_argument(
        "--strip-date",
        required=True,
        help="Experiment date, preferably YYYY-MM-DD.",
    )
    parser.add_argument(
        "--temperature-c",
        required=True,
        type=float,
        help="Ambient temperature in degrees Celsius.",
    )
    parser.add_argument(
        "--humidity-rh",
        required=True,
        type=float,
        help="Relative humidity in percent.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Base output directory. A per-run subdirectory is created inside it.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=900.0,
        help="Recording duration in seconds. Default is 15 minutes.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Raw frame capture rate. Full-resolution raw frames are large.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=1.0,
        help="Camera warmup after locked controls are applied, before logging frames.",
    )
    parser.add_argument("--raw-width", type=int, default=DEFAULT_RAW_WIDTH)
    parser.add_argument("--raw-height", type=int, default=DEFAULT_RAW_HEIGHT)
    return parser.parse_args()


def main() -> None:
    """Run the headless strip recording on a Raspberry Pi."""

    args = parse_args()
    experiment = ExperimentArgs(
        strip_lot=args.strip_lot,
        strip_date=args.strip_date,
        temperature_c=float(args.temperature_c),
        humidity_rh=float(args.humidity_rh),
    )
    settings = build_capture_settings(args)
    Picamera2 = import_picamera2()
    run_dir = build_run_dir(Path(args.out_dir), experiment)

    picam2 = Picamera2()
    camera_config: dict[str, Any] = {}
    frame_records: list[FrameRecord] = []
    interrupted = False

    try:
        camera_config = configure_camera(picam2, settings)
        picam2.start()
        if settings.warmup_s > 0:
            time.sleep(settings.warmup_s)

        print(f"Recording raw frames to {run_dir}")
        print(
            "Locked settings: "
            f"ISO {settings.iso_equivalent}, "
            f"ExposureTime {settings.shutter_us} us, "
            f"AnalogueGain {settings.analogue_gain}, "
            f"AWB {settings.awb_enable}"
        )
        frame_records, interrupted = record_frames(picam2, run_dir, settings)
        if interrupted:
            print("Interrupted; writing metadata for frames captured so far.")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; writing metadata for frames captured so far.")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        write_frames_csv(run_dir, frame_records)
        metadata_path = write_metadata_json(
            run_dir,
            experiment,
            settings,
            camera_config,
            frame_records,
            interrupted=interrupted,
        )

    print(f"Captured {len(frame_records)} raw frames")
    print(f"Wrote metadata: {metadata_path}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    """Convert Picamera2/libcamera/numpy values into JSON-safe containers."""

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
