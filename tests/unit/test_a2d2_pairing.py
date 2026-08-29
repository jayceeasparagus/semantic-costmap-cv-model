from pathlib import Path

import pytest

from semantic_costmap.data import DataPairingError, discover_a2d2_records


def _create_frame(
    sequence_root: Path,
    frame_id: str,
    include_label: bool = True,
) -> None:
    image_dir = sequence_root / "camera/cam_front_center"
    label_dir = sequence_root / "label/cam_front_center"
    lidar_dir = sequence_root / "lidar/cam_front_center"
    for directory in (image_dir, label_dir, lidar_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prefix = "20180807145028"
    (image_dir / f"{prefix}_camera_frontcenter_{frame_id}.png").touch()
    (image_dir / f"{prefix}_camera_frontcenter_{frame_id}.json").touch()
    (lidar_dir / f"{prefix}_lidar_frontcenter_{frame_id}.npz").touch()
    if include_label:
        (label_dir / f"{prefix}_label_frontcenter_{frame_id}.png").touch()


def test_discovers_complete_frames_in_stable_order(tmp_path: Path) -> None:
    sequence_root = tmp_path / "20180807_145028"
    _create_frame(sequence_root, "000000127")
    _create_frame(sequence_root, "000000091")

    records = discover_a2d2_records(
        sequence_root,
        camera="front_center",
        relative_to=tmp_path,
    )

    assert [record.frame_id for record in records] == ["000000091", "000000127"]
    assert records[0].sequence == "20180807_145028"
    assert records[0].camera == "front_center"
    assert records[0].capture_id == "20180807145028"
    assert records[0].image.startswith(
        "20180807_145028/camera/cam_front_center/"
    )
    assert not Path(records[0].image).is_absolute()


def test_rejects_incomplete_sensor_pairs(tmp_path: Path) -> None:
    sequence_root = tmp_path / "20180807_145028"
    _create_frame(sequence_root, "000000091", include_label=False)

    with pytest.raises(DataPairingError, match="missing label"):
        discover_a2d2_records(sequence_root)


def test_rejects_unknown_camera(tmp_path: Path) -> None:
    with pytest.raises(DataPairingError, match="Unsupported A2D2 camera"):
        discover_a2d2_records(tmp_path, camera="roof_camera")
