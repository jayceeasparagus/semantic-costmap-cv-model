"""Discover synchronized RGB, label, LiDAR, and metadata files in A2D2."""

from __future__ import annotations

import re
from pathlib import Path

from .records import A2D2FrameRecord


class DataPairingError(ValueError):
    """Raised when an A2D2 frame is missing or duplicates a sensor modality."""


_CAMERA_TOKEN_TO_NAME = {
    "frontcenter": "front_center",
    "frontleft": "front_left",
    "frontright": "front_right",
    "sideleft": "side_left",
    "sideright": "side_right",
    "rearcenter": "rear_center",
}

_FILENAME_PATTERN = re.compile(
    r"^(?P<capture_id>\d+)_"
    r"(?P<modality>camera|label|lidar)_"
    r"(?P<camera>[a-z]+)_"
    r"(?P<frame_id>\d+)\."
    r"(?P<extension>png|json|npz)$"
)

FrameKey = tuple[str, str]


def discover_a2d2_records(
    sequence_root: str | Path,
    camera: str = "front_center",
    relative_to: str | Path | None = None,
) -> list[A2D2FrameRecord]:
    """Return complete multimodal records from one A2D2 sequence directory.

    Canonical extracted A2D2 sequence directories and the project's compact
    one-frame sample layout are both supported. Unrelated files, including
    Windows Zone.Identifier files, are ignored.
    """
    sequence_root = Path(sequence_root).resolve()
    camera_name, camera_token = _normalize_camera(camera)
    directories = _resolve_directories(sequence_root, camera_name)

    image_files = _index_files(
        directories["image"], "camera", camera_token, "png"
    )
    label_files = _index_files(
        directories["label"], "label", camera_token, "png"
    )
    lidar_files = _index_files(
        directories["lidar"], "lidar", camera_token, "npz"
    )
    metadata_files = _index_files(
        directories["metadata"], "camera", camera_token, "json"
    )

    modality_indices = {
        "image": image_files,
        "label": label_files,
        "lidar": lidar_files,
        "metadata": metadata_files,
    }
    all_keys = set().union(*(set(index) for index in modality_indices.values()))
    if not all_keys:
        raise DataPairingError(f"No A2D2 frames found under {sequence_root}")

    incomplete: list[str] = []
    for key in sorted(all_keys):
        missing = [name for name, index in modality_indices.items() if key not in index]
        if missing:
            incomplete.append(f"{key[0]}:{key[1]} missing {','.join(missing)}")
    if incomplete:
        preview = "; ".join(incomplete[:10])
        remainder = len(incomplete) - 10
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise DataPairingError(f"Incomplete A2D2 frame pairs: {preview}{suffix}")

    path_root = Path(relative_to).resolve() if relative_to is not None else None
    return [
        A2D2FrameRecord(
            sequence=sequence_root.name,
            camera=camera_name,
            capture_id=capture_id,
            frame_id=frame_id,
            image=_serialize_path(image_files[key], path_root),
            label=_serialize_path(label_files[key], path_root),
            lidar=_serialize_path(lidar_files[key], path_root),
            metadata=_serialize_path(metadata_files[key], path_root),
        )
        for key in sorted(all_keys)
        for capture_id, frame_id in (key,)
    ]


def _normalize_camera(camera: str) -> tuple[str, str]:
    token = camera.removeprefix("cam_").replace("_", "").lower()
    try:
        return _CAMERA_TOKEN_TO_NAME[token], token
    except KeyError as error:
        supported = ", ".join(sorted(_CAMERA_TOKEN_TO_NAME.values()))
        raise DataPairingError(
            f"Unsupported A2D2 camera {camera!r}; choose one of: {supported}"
        ) from error


def _resolve_directories(sequence_root: Path, camera_name: str) -> dict[str, Path]:
    camera_directory_name = f"cam_{camera_name}"
    canonical = {
        "image": sequence_root / "camera" / camera_directory_name,
        "label": sequence_root / "label" / camera_directory_name,
        "lidar": sequence_root / "lidar" / camera_directory_name,
    }
    if all(path.is_dir() for path in canonical.values()):
        metadata_directory = canonical["image"]
        return {**canonical, "metadata": metadata_directory}

    compact = {
        "image": sequence_root / "camera",
        "label": sequence_root / "label",
        "lidar": sequence_root / "lidar",
        "metadata": sequence_root / "metadata",
    }
    if all(path.is_dir() for path in compact.values()):
        return compact

    missing_canonical = [
        str(path.relative_to(sequence_root))
        for path in canonical.values()
        if not path.is_dir()
    ]
    raise DataPairingError(
        f"Could not find canonical or compact A2D2 layout under {sequence_root}; "
        f"missing canonical directories={missing_canonical}"
    )


def _index_files(
    directory: Path,
    expected_modality: str,
    expected_camera_token: str,
    expected_extension: str,
) -> dict[FrameKey, Path]:
    index: dict[FrameKey, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = _FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if (
            match["modality"] != expected_modality
            or match["camera"] != expected_camera_token
            or match["extension"] != expected_extension
        ):
            continue

        key = match["capture_id"], match["frame_id"]
        if key in index:
            raise DataPairingError(
                f"Duplicate {expected_modality} file for {key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def _serialize_path(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise DataPairingError(
            f"Cannot store {resolved} relative to manifest root {root}"
        ) from error
