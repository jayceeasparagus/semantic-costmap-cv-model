"""Build and read explicit image/mask manifests for reproducible data splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleRecord:
    image: str
    mask: str
    sequence: str
    frame_id: str


def _frame_id(path: Path) -> str:
    return path.stem.rsplit("_", maxsplit=1)[-1]


def pair_a2d2_directories(
    image_dir: str | Path,
    mask_dir: str | Path,
    sequence: str,
    root: str | Path | None = None,
) -> list[SampleRecord]:
    """Pair camera and converted-mask PNGs by their trailing frame ID."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    root_path = Path(root).resolve() if root is not None else None

    images = {_frame_id(path): path for path in image_dir.glob("*.png")}
    masks = {_frame_id(path): path for path in mask_dir.glob("*.png")}

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Unpaired A2D2 files: "
            f"missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )

    def display_path(path: Path) -> str:
        resolved = path.resolve()
        if root_path is None:
            return str(resolved)
        try:
            return str(resolved.relative_to(root_path))
        except ValueError as error:
            raise ValueError(f"Path is outside manifest root: {resolved}") from error

    return [
        SampleRecord(
            image=display_path(images[frame_id]),
            mask=display_path(masks[frame_id]),
            sequence=sequence,
            frame_id=frame_id,
        )
        for frame_id in sorted(images)
    ]


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> int:
    path = Path(path)
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return len(records)


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(SampleRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid manifest record at line {line_number}: {error}"
                ) from error
    return records
