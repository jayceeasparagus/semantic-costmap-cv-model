from pathlib import Path

import pytest

from semantic_costmap.playback import discover_frame_pairs, summarize_timings


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
