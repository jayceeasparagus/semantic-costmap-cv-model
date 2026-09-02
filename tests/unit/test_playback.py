from pathlib import Path

import pytest

from semantic_costmap.playback import (
    discover_frame_pairs,
    load_pose_csv,
    summarize_timings,
)


def test_discover_frame_pairs_keeps_only_matching_ids(tmp_path: Path):
    images = tmp_path / "camera"
    lidars = tmp_path / "lidar"
    images.mkdir()
    lidars.mkdir()
    (images / "sample_camera_000000002.png").touch()
    (images / "sample_camera_000000001.png").touch()
    (lidars / "sample_lidar_000000001.npz").touch()
    (lidars / "sample_lidar_000000003.npz").touch()

    pairs = discover_frame_pairs(images, lidars)

    assert len(pairs) == 1
    assert pairs[0].frame_id == "000000001"


def test_summarize_timings_reports_mean_and_fps():
    summary = summarize_timings(
        [
            {"inference": 80.0, "total": 100.0},
            {"inference": 100.0, "total": 120.0},
        ]
    )
    assert summary["frame_count"] == 2
    assert summary["stages_ms"]["inference"]["mean"] == 90.0
    assert summary["mean_fps"] == pytest.approx(1000.0 / 110.0)


def test_load_pose_csv_returns_timestamped_map_to_base_poses(tmp_path: Path):
    pose_path = tmp_path / "poses.csv"
    pose_path.write_text(
        "frame_id,timestamp,x,y,yaw\n"
        "1,10.5,2.0,-1.0,0.25\n"
        "000000002,10.6,2.1,-0.9,0.26\n"
    )

    records = load_pose_csv(pose_path)

    assert set(records) == {"000000001", "000000002"}
    assert records["000000001"].timestamp == 10.5
    assert records["000000001"].pose.x == 2.0
    assert records["000000001"].pose.yaw == 0.25


def test_load_pose_csv_rejects_missing_pose_fields(tmp_path: Path):
    pose_path = tmp_path / "poses.csv"
    pose_path.write_text("frame_id,x,y\n1,0.0,0.0\n")

    with pytest.raises(ValueError, match="missing columns"):
        load_pose_csv(pose_path)
