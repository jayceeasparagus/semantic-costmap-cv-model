#!/usr/bin/env python3
"""Run semantic segmentation and paint independently projected A2D2 points."""

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from semantic_costmap.config import (
    DEFAULT_CHECKPOINT_PATH,
    SEMANTIC_CLASSES,
    class_colors,
)
from semantic_costmap.fusion import paint_points, save_painted_cloud
from semantic_costmap.geometry import (
    load_a2d2_calibration,
    project_camera_points,
    transform_between_views,
    transform_points,
)
from semantic_costmap.inference import SemanticSegmenter


DEFAULT_IMAGE = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)
DEFAULT_LIDAR = Path(
    "data/raw/a2d2_sample/lidar/"
    "20180807145028_lidar_frontcenter_000000091.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--lidar", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/a2d2_cams_lidars.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    lidar = np.load(args.lidar)
    calibration = load_a2d2_calibration(args.calibration)
    if image.size != calibration.resolution:
        raise ValueError("image resolution does not match camera calibration")

    start = time.perf_counter()
    segmenter = SemanticSegmenter(args.checkpoint, args.device)
    segmentation = segmenter.predict(image)
    inference_seconds = time.perf_counter() - start

    projection = project_camera_points(
        lidar["points"],
        calibration.camera_matrix,
        calibration.resolution,
    )
    camera_to_vehicle = transform_between_views(
        calibration.view,
        calibration.vehicle_view,
    )
    points_vehicle = transform_points(lidar["points"], camera_to_vehicle)
    painted = paint_points(
        lidar["points"],
        points_vehicle,
        projection,
        segmentation,
        lidar_ids=lidar["lidar_id"] if "lidar_id" in lidar else None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = args.output_dir / "painted_points.npz"
    overlay_path = args.output_dir / "painted_points_overlay.png"
    save_painted_cloud(
        cloud_path,
        painted,
        raytrace_origin_vehicle=calibration.origin_vehicle,
    )

    palette = np.asarray(class_colors(), dtype=np.uint8)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row, column, class_id in zip(
        painted.rows,
        painted.columns,
        painted.class_ids,
    ):
        color = tuple(int(value) for value in palette[class_id])
        draw.ellipse(
            (column - 2, row - 2, column + 2, row + 2),
            fill=color,
        )
    overlay.save(overlay_path)

    print("Projection source: 3D points + calibrated camera matrix")
    print(f"Checkpoint epoch: {segmenter.epoch}")
    print(f"Painted points: {len(painted.class_ids)}")
    print(f"Mean point confidence: {painted.confidence.mean():.4f}")
    print(f"CPU/GPU inference time: {inference_seconds:.3f} s")
    print(f"Saved: {cloud_path}")
    print(f"Saved: {overlay_path}")
    for semantic_class in SEMANTIC_CLASSES:
        count = np.count_nonzero(painted.class_ids == semantic_class.class_id)
        print(f"{semantic_class.name}: {count}")


if __name__ == "__main__":
    main()
