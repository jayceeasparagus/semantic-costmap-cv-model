#!/usr/bin/env python3
"""Discover complete A2D2 frames and write a reproducible JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_costmap.data import discover_a2d2_records, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        action="append",
        required=True,
        help="A2D2 sequence directory; repeat for multiple sequences",
    )
    parser.add_argument("--camera", default="front_center")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Store manifest paths relative to this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for sequence_root in args.sequence_root:
        discovered = discover_a2d2_records(
            sequence_root=sequence_root,
            camera=args.camera,
            relative_to=args.path_root,
        )
        records.extend(discovered)
        print(f"{sequence_root}: {len(discovered)} complete frame(s)")

    count = write_manifest(records, args.output)
    sequence_count = len({record.sequence for record in records})
    print(f"Wrote {count} frame(s) from {sequence_count} sequence(s)")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
