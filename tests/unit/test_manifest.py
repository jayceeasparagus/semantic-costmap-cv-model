from offroad_vision.data.manifest import (
    pair_a2d2_directories,
    read_manifest,
    write_manifest,
)


def test_manifest_pairs_by_frame_id(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    for frame_id in ("000000091", "000000127"):
        (image_dir / f"sequence_camera_frontcenter_{frame_id}.png").touch()
        (mask_dir / f"sequence_trainid_frontcenter_{frame_id}.png").touch()

    records = pair_a2d2_directories(
        image_dir,
        mask_dir,
        sequence="sequence",
        root=tmp_path,
    )
    manifest_path = tmp_path / "split.jsonl"
    count = write_manifest(records, manifest_path)

    assert count == 2
    assert read_manifest(manifest_path) == records
    assert records[0].frame_id == "000000091"
    assert records[0].image.startswith("images/")
