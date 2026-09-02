#!/usr/bin/env python3
"""Run semantic segmentation on one RGB image."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_costmap.config import (
    DEFAULT_CHECKPOINT_PATH,
    SEMANTIC_CLASSES,
)
from semantic_costmap.inference import SemanticSegmenter, colorize_class_ids


DEFAULT_IMAGE_PATH = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    image = Image.open(args.image).convert("RGB")
    segmenter = SemanticSegmenter(args.checkpoint, args.device)
    result = segmenter.predict(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    id_path = args.output_dir / f"{stem}_class_ids.png"
    color_path = args.output_dir / f"{stem}_classes.png"
    overlay_path = args.output_dir / f"{stem}_overlay.png"
    confidence_path = args.output_dir / f"{stem}_confidence.png"

    color_mask = Image.fromarray(colorize_class_ids(result.class_ids))
    Image.fromarray(result.class_ids).save(id_path)
    color_mask.save(color_path)
    Image.blend(image, color_mask, alpha=0.45).save(overlay_path)
    Image.fromarray((result.confidence * 255).astype(np.uint8)).save(confidence_path)

    print("Device:", segmenter.device)
    print("Checkpoint epoch:", segmenter.epoch)
    print("Saved outputs to:", args.output_dir)
    for semantic_class in SEMANTIC_CLASSES:
        percentage = np.mean(result.class_ids == semantic_class.class_id) * 100.0
        print(f"{semantic_class.name}: {percentage:.2f}%")


if __name__ == "__main__":
    main()
