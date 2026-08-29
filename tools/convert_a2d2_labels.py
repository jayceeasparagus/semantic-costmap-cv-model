#!/usr/bin/env python3
"""Convert a directory of A2D2 RGB labels into compact training masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from offroad_vision.data.a2d2_labels import A2D2LabelConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument(
        "--class-list",
        type=Path,
        default=PROJECT_ROOT / "configs/a2d2_class_list.json",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "configs/a2d2_semantic_mapping_v2.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = A2D2LabelConverter.from_files(args.class_list, args.mapping)
    input_paths = sorted(args.input_dir.rglob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(
            f"No files matched {args.pattern!r} under {args.input_dir}"
        )

    counts = np.zeros(256, dtype=np.int64)
    for input_path in input_paths:
        relative_path = input_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        train_ids = converter.convert_file(input_path, output_path)
        counts += np.bincount(train_ids.reshape(-1), minlength=256)

    print(f"Converted masks: {len(input_paths)}")
    print(f"Output directory: {args.output_dir}")
    print("Pixel distribution:")
    for train_id, class_name in sorted(
        converter.taxonomy.train_id_to_name.items()
    ):
        print(f"  {train_id}: {class_name:<18} {counts[train_id]:>12} pixels")
    print(
        f"  {converter.taxonomy.ignore_id}: {'ignore':<18} "
        f"{counts[converter.taxonomy.ignore_id]:>12} pixels"
    )


if __name__ == "__main__":
    main()
