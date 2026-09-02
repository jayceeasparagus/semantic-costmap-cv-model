"""Timestamp-aligned planar odometry from A2D2 bus signals.

The A2D2 bus file contains one record for each front-center camera frame.  A
record also contains short, timestamped signal histories.  This module
flattens those histories, interpolates vehicle speed and yaw rate at each
camera timestamp, and integrates a simple unicycle model.  The result is
odometry for replay and debugging; it is not SLAM ground truth.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FRAME_ID_PATTERN = re.compile(r"_(\d{9})\.json$")
DEFAULT_SPEED_SIGNAL = "vehicle_speed"
DEFAULT_YAW_RATE_SIGNAL = "angular_velocity_omega_z"


@dataclass(frozen=True)
class SignalSeries:
    """One bus signal represented as sorted, de-duplicated samples."""

    timestamps_s: np.ndarray
    values: np.ndarray
    unit: str | None

    def interpolate(self, timestamp_s: float) -> float:
        if self.timestamps_s.size == 0:
            raise ValueError("cannot interpolate an empty signal")
        if timestamp_s < self.timestamps_s[0] or timestamp_s > self.timestamps_s[-1]:
            raise ValueError(
                f"timestamp {timestamp_s:.6f} is outside signal range "
                f"[{self.timestamps_s[0]:.6f}, {self.timestamps_s[-1]:.6f}]"
            )
        return float(np.interp(timestamp_s, self.timestamps_s, self.values))


@dataclass(frozen=True)
class BusFrame:
    """A2D2 bus record associated with one camera frame."""

    frame_id: str
    frame_name: str
    timestamp_s: float
    signals: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class OdometryRecord:
    frame_id: str
    timestamp: float
    x: float
    y: float
    yaw: float


def _seconds(timestamp: float | int) -> float:
    """Convert A2D2 integer microseconds to seconds, preserving test seconds."""

    value = float(timestamp)
    return value / 1_000_000.0 if abs(value) > 1e11 else value


def _frame_id_from_name(name: str) -> str:
    match = FRAME_ID_PATTERN.search(Path(name).name)
    if match is None:
        raise ValueError(f"could not find a nine-digit frame ID in {name!r}")
    return match.group(1)


def load_bus_frames(path: str | Path) -> list[BusFrame]:
    """Read the actual A2D2 list-of-frame-records bus format."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("A2D2 bus JSON must contain a list of frame records")

    frames: list[BusFrame] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        frame_name = str(record.get("frame_name", ""))
        if not frame_name or "timestamp" not in record:
            continue
        flexray = record.get("flexray", {})
        if not isinstance(flexray, dict):
            flexray = {}
        frames.append(
            BusFrame(
                frame_id=_frame_id_from_name(frame_name),
                frame_name=frame_name,
                timestamp_s=_seconds(record["timestamp"]),
                signals=flexray,
            )
        )
    if not frames:
        raise ValueError("A2D2 bus JSON contains no usable camera frame records")
    return frames


def _series_from_frames(frames: Iterable[BusFrame], signal_name: str) -> SignalSeries:
    timestamps: list[float] = []
    values: list[float] = []
    unit: str | None = None
    for frame in frames:
        sample = frame.signals.get(signal_name)
        if not isinstance(sample, dict):
            continue
        raw_timestamps = sample.get("timestamps", [])
        raw_values = sample.get("values", [])
        if len(raw_timestamps) != len(raw_values):
            raise ValueError(f"signal {signal_name!r} has mismatched timestamps/values")
        unit = unit or sample.get("unit")
        timestamps.extend(_seconds(item) for item in raw_timestamps)
        values.extend(float(item) for item in raw_values)

    if not timestamps:
        raise ValueError(f"signal {signal_name!r} was not found in the bus data")
    order = np.argsort(np.asarray(timestamps, dtype=np.float64), kind="stable")
    sorted_timestamps = np.asarray(timestamps, dtype=np.float64)[order]
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    unique_timestamps, unique_indices = np.unique(sorted_timestamps, return_index=True)
    return SignalSeries(unique_timestamps, sorted_values[unique_indices], unit)


def _convert_speed(value: float, unit: str | None) -> float:
    normalized = (unit or "").lower()
    if "kilo" in normalized or "km" in normalized:
        return value / 3.6
    if "mile" in normalized or "mph" in normalized:
        return value * 0.44704
    return value


def _convert_yaw_rate(value: float, unit: str | None) -> float:
    normalized = (unit or "").lower()
    if "degre" in normalized or "degree" in normalized or "deg" in normalized:
        return math.radians(value)
    return value


def _camera_timestamps(
    frames: list[BusFrame], metadata_directory: str | Path | None
) -> list[tuple[str, float]]:
    if metadata_directory is None:
        return [(frame.frame_id, frame.timestamp_s) for frame in frames]

    bus_ids = {frame.frame_id for frame in frames}
    records: list[tuple[str, float]] = []
    for path in sorted(Path(metadata_directory).rglob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "cam_tstamp" not in payload:
            continue
        frame_id = _frame_id_from_name(path.name)
        if frame_id in bus_ids:
            records.append((frame_id, _seconds(payload["cam_tstamp"])))
    if not records:
        raise ValueError("camera metadata contains no frames matching the bus JSON")
    return sorted(set(records), key=lambda item: item[1])


def build_odometry(
    frames: list[BusFrame],
    metadata_directory: str | Path | None = None,
    *,
    speed_signal: str = DEFAULT_SPEED_SIGNAL,
    yaw_rate_signal: str = DEFAULT_YAW_RATE_SIGNAL,
    initial_yaw: float = 0.0,
    initial_x: float = 0.0,
    initial_y: float = 0.0,
) -> list[OdometryRecord]:
    """Integrate speed and yaw rate at camera timestamps.

    The planar convention is x forward, y left, and positive yaw counter-
    clockwise.  Timestamp gaps are integrated directly; a negative or zero
    gap is treated as a repeated timestamp and contributes no motion.
    """

    if len(frames) == 0:
        raise ValueError("at least one bus frame is required")
    speed = _series_from_frames(frames, speed_signal)
    yaw_rate = _series_from_frames(frames, yaw_rate_signal)
    camera_frames = _camera_timestamps(frames, metadata_directory)

    result: list[OdometryRecord] = []
    x = float(initial_x)
    y = float(initial_y)
    yaw = float(initial_yaw)
    previous_time: float | None = None
    for frame_id, timestamp_s in camera_frames:
        if previous_time is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp_s - previous_time)
        speed_mps = _convert_speed(speed.interpolate(timestamp_s), speed.unit)
        yaw_rate_rps = _convert_yaw_rate(
            yaw_rate.interpolate(timestamp_s), yaw_rate.unit
        )
        yaw_midpoint = yaw + 0.5 * yaw_rate_rps * dt
        x += speed_mps * math.cos(yaw_midpoint) * dt
        y += speed_mps * math.sin(yaw_midpoint) * dt
        yaw = _wrap_angle(yaw + yaw_rate_rps * dt)
        result.append(OdometryRecord(frame_id, timestamp_s, x, y, yaw))
        previous_time = timestamp_s
    return result


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def write_pose_csv(records: Iterable[OdometryRecord], path: str | Path) -> None:
    """Write the format consumed by offline playback and map accumulation."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot write an empty pose CSV")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_id", "timestamp", "x", "y", "yaw"))
        writer.writerows(
            (row.frame_id, f"{row.timestamp:.9f}", f"{row.x:.9f}", f"{row.y:.9f}", f"{row.yaw:.9f}")
            for row in rows
        )

