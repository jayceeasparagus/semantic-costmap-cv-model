"""Reusable end-to-end RGB and LiDAR processing pipeline."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

from semantic_costmap.costmap import (
    CostmapConfig,
    SemanticCostmap,
    build_semantic_costmap,
)
from semantic_costmap.fusion import PaintedPointCloud, paint_points
from semantic_costmap.geometry import (
    CameraCalibration,
    load_a2d2_calibration,
    project_camera_points,
    transform_between_views,
    transform_points,
)
from semantic_costmap.inference import SegmentationResult, SemanticSegmenter


@dataclass(frozen=True)
class FrameResult:
    image: Image.Image
    segmentation: SegmentationResult
    painted_points: PaintedPointCloud
    costmap: SemanticCostmap
    timings_ms: dict[str, float]


class SemanticCostmapPipeline:
    """Keep the model and calibration loaded while processing many frames."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        calibration_path: str | Path,
        device: str = "auto",
        costmap_config: CostmapConfig | None = None,
    ) -> None:
        self.segmenter = SemanticSegmenter(checkpoint_path, device)
        self.calibration: CameraCalibration = load_a2d2_calibration(
            calibration_path
        )
        self.costmap_config = costmap_config or CostmapConfig()
        self.camera_to_vehicle = transform_between_views(
            self.calibration.view,
            self.calibration.vehicle_view,
        )

    def process(self, image_path: str | Path, lidar_path: str | Path) -> FrameResult:
        start_total = time.perf_counter()
        start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        lidar = np.load(lidar_path)
        load_ms = (time.perf_counter() - start) * 1000.0
        if image.size != self.calibration.resolution:
            raise ValueError("image resolution does not match calibration")

        start = time.perf_counter()
        segmentation = self.segmenter.predict(image)
        inference_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        projection = project_camera_points(
            lidar["points"],
            self.calibration.camera_matrix,
            self.calibration.resolution,
        )
        points_vehicle = transform_points(
            lidar["points"],
            self.camera_to_vehicle,
        )
        projection_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        painted = paint_points(
            lidar["points"],
            points_vehicle,
            projection,
            segmentation,
        )
        fusion_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        costmap = build_semantic_costmap(
            painted,
            self.costmap_config,
            raw_points_vehicle=points_vehicle,
        )
        costmap_ms = (time.perf_counter() - start) * 1000.0
        total_ms = (time.perf_counter() - start_total) * 1000.0

        return FrameResult(
            image=image,
            segmentation=segmentation,
            painted_points=painted,
            costmap=costmap,
            timings_ms={
                "load": load_ms,
                "inference": inference_ms,
                "projection": projection_ms,
                "fusion": fusion_ms,
                "costmap": costmap_ms,
                "total": total_ms,
            },
        )
