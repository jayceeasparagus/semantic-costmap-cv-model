"""A2D2 sensor calibration loading and rigid transforms."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration values needed by the front-camera pipeline."""

    name: str
    camera_matrix: np.ndarray
    resolution: tuple[int, int]
    view: dict
    vehicle_view: dict
    origin_vehicle: np.ndarray


def view_axes(view: dict) -> np.ndarray:
    """Return an orthonormal view-to-vehicle rotation matrix."""

    x_axis = np.asarray(view["x-axis"], dtype=np.float64)
    y_axis = np.asarray(view["y-axis"], dtype=np.float64)

    if np.linalg.norm(x_axis) < 1e-12 or np.linalg.norm(y_axis) < 1e-12:
        raise ValueError("view axes must have non-zero length")

    x_axis /= np.linalg.norm(x_axis)
    y_axis -= x_axis * np.dot(y_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-12:
        raise ValueError("view axes must not be parallel")
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    return np.column_stack((x_axis, y_axis, z_axis))


def transform_to_vehicle(view: dict) -> np.ndarray:
    """Build the homogeneous transform from a sensor view to the vehicle."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = view_axes(view)
    transform[:3, 3] = np.asarray(view["origin"], dtype=np.float64)
    return transform


def transform_between_views(source_view: dict, target_view: dict) -> np.ndarray:
    """Return a transform that maps source-view points into target-view points."""

    source_to_vehicle = transform_to_vehicle(source_view)
    target_to_vehicle = transform_to_vehicle(target_view)
    return np.linalg.inv(target_to_vehicle) @ source_to_vehicle


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an Nx3 point array."""

    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")

    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def load_a2d2_calibration(
    path: str | Path,
    camera_name: str = "front_center",
) -> CameraCalibration:
    """Load one camera and the vehicle frame from A2D2 calibration JSON."""

    calibration_path = Path(path)
    data = json.loads(calibration_path.read_text())
    try:
        camera = data["cameras"][camera_name]
        vehicle_view = data["vehicle"]["view"]
    except KeyError as error:
        raise ValueError(
            f"missing A2D2 calibration field: {error.args[0]}"
        ) from error

    matrix = np.asarray(camera["CamMatrix"], dtype=np.float64)
    resolution = tuple(int(value) for value in camera["Resolution"])
    if matrix.shape != (3, 3) or len(resolution) != 2:
        raise ValueError("invalid camera matrix or resolution")

    return CameraCalibration(
        name=camera_name,
        camera_matrix=matrix,
        resolution=resolution,
        view=camera["view"],
        vehicle_view=vehicle_view,
        origin_vehicle=transform_to_vehicle(camera["view"])[:3, 3].copy(),
    )
