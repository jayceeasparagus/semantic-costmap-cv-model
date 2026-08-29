#!/usr/bin/env python3
"""Pair prepared A2D2 images and masks into a JSON-lines manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from offroad_vision.data.manifest import pair_a2d2_directories, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Store paths relative to this project/data root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = pair_a2d2_directories(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        sequence=args.sequence,
        root=args.root,
    )
    count = write_manifest(records, args.output)
    print(f"Manifest: {args.output}")
    print(f"Sequence: {args.sequence}")
    print(f"Paired samples: {count}")


if __name__ == "__main__":
    main()
