import json
from pathlib import Path

import pytest

from semantic_costmap.odometry import build_odometry, load_bus_frames, write_pose_csv
from semantic_costmap.playback import load_pose_csv


def make_bus(path: Path) -> None:
    records = []
    for frame_id, timestamp in (("000000001", 0.0), ("000000002", 1.0), ("000000003", 2.0)):
        records.append(
            {
                "frame_name": f"sequence_camera_frontcenter_{frame_id}.json",
                "timestamp": timestamp,
                "flexray": {
                    "vehicle_speed": {
                        "timestamps": [0.0, 1.0, 2.0],
                        "values": [3.6, 3.6, 3.6],
                        "unit": "km/h",
                    },
                    "angular_velocity_omega_z": {
                        "timestamps": [0.0, 1.0, 2.0],
                        "values": [0.0, 90.0, 0.0],
                        "unit": "deg/s",
                    },
                },
            }
        )
    path.write_text(json.dumps(records))


def test_load_and_integrate_bus_signals_with_units(tmp_path: Path):
    bus_path = tmp_path / "bus.json"
    make_bus(bus_path)

    records = build_odometry(load_bus_frames(bus_path))

    assert [item.frame_id for item in records] == ["000000001", "000000002", "000000003"]
    assert records[1].x == pytest.approx(2**-0.5)
    assert records[1].y == pytest.approx(2**-0.5)
    assert records[1].yaw == pytest.approx(3.141592653589793 / 2.0)
    assert records[2].yaw == pytest.approx(3.141592653589793 / 2.0)


def test_metadata_timestamps_are_used(tmp_path: Path):
    bus_path = tmp_path / "bus.json"
    make_bus(bus_path)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for frame_id, timestamp in (("000000001", 0.0), ("000000002", 0.5), ("000000003", 1.0)):
        (metadata / f"sequence_camera_frontcenter_{frame_id}.json").write_text(
            json.dumps({"cam_tstamp": timestamp})
        )

    records = build_odometry(load_bus_frames(bus_path), metadata)

    assert records[1].timestamp == pytest.approx(0.5)
    assert records[1].x == pytest.approx(0.4903926402)


def test_aggregate_bus_format_uses_camera_metadata(tmp_path: Path):
    bus_path = tmp_path / "aggregate_bus.json"
    bus_path.write_text(
        json.dumps(
            {
                "vehicle_speed": {
                    "unit": "Unit_KiloMeterPerHour",
                    "values": [[1533906414000000, 3.6], [1533906415000000, 3.6]],
                },
                "angular_velocity_omega_z": {
                    "unit": "Unit_DegreOfArcPerSecon",
                    "values": [[1533906414000000, 0.0], [1533906415000000, 0.0]],
                },
            }
        )
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for frame_id, timestamp in (("000000001", 1.0), ("000000002", 2.0)):
        (metadata / f"sequence_camera_frontcenter_{frame_id}.json").write_text(
            json.dumps({"cam_tstamp": (1533906413 + timestamp) * 1_000_000})
        )

    records = build_odometry(load_bus_frames(bus_path), metadata)

    assert len(records) == 2
    assert records[1].x == pytest.approx(1.0)


def test_pose_csv_round_trip(tmp_path: Path):
    bus_path = tmp_path / "bus.json"
    make_bus(bus_path)
    pose_path = tmp_path / "poses.csv"
    write_pose_csv(build_odometry(load_bus_frames(bus_path)), pose_path)

    loaded = load_pose_csv(pose_path)

    assert loaded["000000003"].pose.x == pytest.approx(2**-0.5)


def test_missing_signal_is_explicit(tmp_path: Path):
    path = tmp_path / "bus.json"
    path.write_text(
        json.dumps(
            [{
                "frame_name": "sequence_camera_frontcenter_000000001.json",
                "timestamp": 0.0,
                "flexray": {},
            }]
        )
    )
    with pytest.raises(ValueError, match="vehicle_speed"):
        build_odometry(load_bus_frames(path))
