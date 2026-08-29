#!/usr/bin/env python3
"""Validate that the compact taxonomy covers every A2D2 label exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from offroad_vision.data.taxonomy import SemanticTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    class_list = json.loads(args.class_list.read_text(encoding="utf-8"))
    taxonomy = SemanticTaxonomy.from_yaml(args.mapping)
    taxonomy.validate(class_list.values())

    print(f"Valid taxonomy version: {taxonomy.version}")
    print(f"A2D2 source classes covered: {len(class_list)}")
    print(f"Model output classes: {taxonomy.num_classes}")
    print(f"Ignore ID: {taxonomy.ignore_id}")
    for group in sorted(taxonomy.groups, key=lambda item: item.train_id):
        cost = "no cost" if group.costmap_cost is None else group.costmap_cost
        print(
            f"  {group.train_id}: {group.name:<18} "
            f"source classes={len(group.classes):>2}, cost={cost}"
        )
    print(f"Ignored source classes: {len(taxonomy.ignored_classes)}")


if __name__ == "__main__":
    main()
