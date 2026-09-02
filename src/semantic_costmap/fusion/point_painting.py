"""Attach image semantic probabilities to geometrically projected points."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_costmap.config import class_costs
from semantic_costmap.geometry import ProjectionResult
from semantic_costmap.inference import SegmentationResult


@dataclass(frozen=True)
class PaintedPointCloud:
    """LiDAR returns enriched with camera semantics."""

    points_camera: np.ndarray
    points_vehicle: np.ndarray
    probabilities: np.ndarray
    class_ids: np.ndarray
    confidence: np.ndarray
    costs: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    source_indices: np.ndarray
    lidar_ids: np.ndarray | None = None
    raytrace_origin_vehicle: np.ndarray | None = None


def paint_points(
    points_camera: np.ndarray,
    points_vehicle: np.ndarray,
    projection: ProjectionResult,
    segmentation: SegmentationResult,
    lidar_ids: np.ndarray | None = None,
) -> PaintedPointCloud:
    """Sample segmentation probabilities at each valid projected LiDAR point."""

    points_camera = np.asarray(points_camera, dtype=np.float64)
    points_vehicle = np.asarray(points_vehicle, dtype=np.float64)
    if points_camera.shape != points_vehicle.shape:
        raise ValueError("camera and vehicle point arrays must have equal shape")
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError("point arrays must have shape (N, 3)")
    if len(projection.valid) != len(points_camera):
        raise ValueError("projection length must match point count")
    if segmentation.class_ids.ndim != 2:
        raise ValueError("segmentation class IDs must have shape (H, W)")
    if segmentation.probabilities.ndim != 3:
        raise ValueError("segmentation probabilities must have shape (C, H, W)")
    if lidar_ids is not None:
        lidar_ids = np.asarray(lidar_ids)
        if lidar_ids.ndim != 1 or len(lidar_ids) != len(points_camera):
            raise ValueError("lidar_ids must have one value per input point")

    height, width = segmentation.class_ids.shape
    valid = projection.valid.copy()
    source_indices = np.flatnonzero(valid)
    rows = np.rint(projection.rows[valid]).astype(np.int32)
    columns = np.rint(projection.columns[valid]).astype(np.int32)
    rows = np.clip(rows, 0, height - 1)
    columns = np.clip(columns, 0, width - 1)

    probabilities = segmentation.probabilities[:, rows, columns].T
    class_ids = probabilities.argmax(axis=1).astype(np.uint8)
    confidence = probabilities.max(axis=1).astype(np.float32)
    costs = np.asarray(class_costs(), dtype=np.int16)[class_ids]

    return PaintedPointCloud(
        points_camera=points_camera[valid],
        points_vehicle=points_vehicle[valid],
        probabilities=probabilities.astype(np.float32),
        class_ids=class_ids,
        confidence=confidence,
        costs=costs,
        rows=rows,
        columns=columns,
        source_indices=source_indices,
        lidar_ids=None if lidar_ids is None else lidar_ids[valid],
    )

def save_painted_cloud(
    path: str | Path,
    cloud: PaintedPointCloud,
    raytrace_origin_vehicle: np.ndarray | None = None,
) -> None:
    """Save the reusable painted point representation as compressed NumPy data."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        points_camera=cloud.points_camera,
        points_vehicle=cloud.points_vehicle,
        probabilities=cloud.probabilities,
        class_ids=cloud.class_ids,
        confidence=cloud.confidence,
        costs=cloud.costs,
        rows=cloud.rows,
        columns=cloud.columns,
        source_indices=cloud.source_indices,
    )
    if cloud.lidar_ids is not None:
        arrays["lidar_ids"] = cloud.lidar_ids
    if raytrace_origin_vehicle is not None:
        origin = np.asarray(raytrace_origin_vehicle, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("raytrace_origin_vehicle must have shape (3,)")
        arrays["raytrace_origin_vehicle"] = origin
    np.savez_compressed(output_path, **arrays)
