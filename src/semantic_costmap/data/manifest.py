"""Read, write, and split reproducible A2D2 JSON-lines manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .records import A2D2FrameRecord


class ManifestError(ValueError):
    """Raised when a manifest is invalid or a sequence split leaks data."""


def write_manifest(
    records: Iterable[A2D2FrameRecord],
    path: str | Path,
) -> int:
    """Atomically write sorted records and return the record count."""
    path = Path(path)
    ordered_records = sorted(records, key=lambda record: record.key)
    _validate_unique_keys(ordered_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in ordered_records:
                output.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        temporary_path.replace(path)
    except OSError as error:
        raise ManifestError(f"Could not write manifest {path}: {error}") from error
    return len(ordered_records)


def read_manifest(path: str | Path) -> list[A2D2FrameRecord]:
    path = Path(path)
    records: list[A2D2FrameRecord] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    values = json.loads(line)
                    if not isinstance(values, dict):
                        raise ValueError("record is not a JSON object")
                    records.append(A2D2FrameRecord.from_dict(values))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ManifestError(
                        f"Invalid record in {path} at line {line_number}: {error}"
                    ) from error
    except OSError as error:
        raise ManifestError(f"Could not read manifest {path}: {error}") from error

    _validate_unique_keys(records)
    return records


def split_by_sequences(
    records: Iterable[A2D2FrameRecord],
    train_sequences: Iterable[str],
    validation_sequences: Iterable[str],
    require_all_sequences: bool = True,
) -> tuple[list[A2D2FrameRecord], list[A2D2FrameRecord]]:
    """Partition complete sequences so neighboring frames cannot leak."""
    records = list(records)
    train_names = set(train_sequences)
    validation_names = set(validation_sequences)
    overlap = sorted(train_names & validation_names)
    if overlap:
        raise ManifestError(f"Sequences assigned to both splits: {overlap}")

    available = {record.sequence for record in records}
    requested = train_names | validation_names
    unknown = sorted(requested - available)
    if unknown:
        raise ManifestError(f"Requested sequences are absent from records: {unknown}")
    if require_all_sequences:
        unassigned = sorted(available - requested)
        if unassigned:
            raise ManifestError(f"Sequences are not assigned to a split: {unassigned}")

    train = [record for record in records if record.sequence in train_names]
    validation = [
        record for record in records if record.sequence in validation_names
    ]
    return train, validation


def _validate_unique_keys(records: Iterable[A2D2FrameRecord]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    duplicates: list[tuple[str, str, str, str]] = []
    for record in records:
        if record.key in seen:
            duplicates.append(record.key)
        seen.add(record.key)
    if duplicates:
        raise ManifestError(f"Duplicate frame keys: {sorted(set(duplicates))}")
