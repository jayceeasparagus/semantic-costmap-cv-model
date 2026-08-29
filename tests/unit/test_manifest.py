from pathlib import Path

import pytest

from semantic_costmap.data import (
    A2D2FrameRecord,
    ManifestError,
    read_manifest,
    split_by_sequences,
    write_manifest,
)


def _record(sequence: str, frame_id: str) -> A2D2FrameRecord:
    prefix = f"data/{sequence}/{frame_id}"
    return A2D2FrameRecord(
        sequence=sequence,
        camera="front_center",
        capture_id="20180807145028",
        frame_id=frame_id,
        image=f"{prefix}.png",
        label=f"{prefix}_label.png",
        lidar=f"{prefix}.npz",
        metadata=f"{prefix}.json",
    )


def test_manifest_round_trip_is_sorted(tmp_path: Path) -> None:
    records = [
        _record("sequence_b", "0002"),
        _record("sequence_a", "0001"),
    ]
    manifest_path = tmp_path / "records.jsonl"

    count = write_manifest(records, manifest_path)
    loaded = read_manifest(manifest_path)

    assert count == 2
    assert loaded == [records[1], records[0]]


def test_sequence_split_keeps_sequences_disjoint() -> None:
    records = [
        _record("train_drive", "0001"),
        _record("train_drive", "0002"),
        _record("validation_drive", "0001"),
    ]

    train, validation = split_by_sequences(
        records,
        train_sequences={"train_drive"},
        validation_sequences={"validation_drive"},
    )

    assert {record.sequence for record in train} == {"train_drive"}
    assert {record.sequence for record in validation} == {"validation_drive"}
    assert {record.key for record in train}.isdisjoint(
        {record.key for record in validation}
    )


def test_sequence_split_rejects_leakage() -> None:
    records = [_record("shared_drive", "0001")]

    with pytest.raises(ManifestError, match="both splits"):
        split_by_sequences(
            records,
            train_sequences={"shared_drive"},
            validation_sequences={"shared_drive"},
        )


def test_manifest_rejects_duplicate_frame_keys(tmp_path: Path) -> None:
    record = _record("sequence", "0001")

    with pytest.raises(ManifestError, match="Duplicate frame keys"):
        write_manifest([record, record], tmp_path / "duplicates.jsonl")
