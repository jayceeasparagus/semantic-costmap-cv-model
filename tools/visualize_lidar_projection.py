#!/usr/bin/env python3
"""Paint A2D2 LiDAR points using the dataset-provided pixel projection."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from semantic_costmap.config import (
    DEFAULT_CHECKPOINT_PATH,
    SEMANTIC_CLASSES,
    class_colors,
    class_costs,
)
from semantic_costmap.inference import SemanticSegmenter


DEFAULT_IMAGE_PATH = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)
DEFAULT_LIDAR_PATH = Path(
    "data/raw/a2d2_sample/lidar/"
    "20180807145028_lidar_frontcenter_000000091.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--lidar", type=Path, default=DEFAULT_LIDAR_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    lidar = np.load(args.lidar)
    segmenter = SemanticSegmenter(args.checkpoint, args.device)
    prediction = segmenter.predict(image)

    rows = np.rint(lidar["row"]).astype(np.int32)
    cols = np.rint(lidar["col"]).astype(np.int32)
    valid = (
        (rows >= 0)
        & (rows < image.height)
        & (cols >= 0)
        & (cols < image.width)
    )
    rows = rows[valid]
    cols = cols[valid]
    predicted_classes = prediction.class_ids[rows, cols]
    predicted_probabilities = prediction.probabilities[:, rows, cols].T
    predicted_confidence = prediction.confidence[rows, cols]
    costs = np.asarray(class_costs(), dtype=np.int16)[predicted_classes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_path = args.output_dir / "semantic_lidar_points.npz"
    overlay_path = args.output_dir / "semantic_lidar_overlay.png"
    np.savez_compressed(
        point_path,
        points=lidar["points"][valid],
        row=rows,
        col=cols,
        depth=lidar["depth"][valid],
        lidar_id=lidar["lidar_id"][valid],
        predicted_class=predicted_classes,
        predicted_probability=predicted_probabilities,
        predicted_confidence=predicted_confidence,
        predicted_cost=costs,
    )

    palette = np.asarray(class_colors(), dtype=np.uint8)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row, col, class_id in zip(rows, cols, predicted_classes):
        color = tuple(int(value) for value in palette[class_id])
        draw.ellipse((col - 2, row - 2, col + 2, row + 2), fill=color)
    overlay.save(overlay_path)

    print("Projection source: A2D2 row/col baseline")
    print("Checkpoint epoch:", segmenter.epoch)
    print("Valid LiDAR points:", len(rows))
    print("Mean point confidence:", float(predicted_confidence.mean()))
    print("Saved:", point_path)
    print("Saved:", overlay_path)
    for semantic_class in SEMANTIC_CLASSES:
        count = np.count_nonzero(predicted_classes == semantic_class.class_id)
        print(f"{semantic_class.name}: {count}")


if __name__ == "__main__":
    main()
