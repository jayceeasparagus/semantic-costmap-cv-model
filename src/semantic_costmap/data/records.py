"""Typed records shared by A2D2 discovery and training manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class A2D2FrameRecord:
    """Paths and identity for one synchronized A2D2 sensor frame."""

    sequence: str
    camera: str
    capture_id: str
    frame_id: str
    image: str
    label: str
    lidar: str
    metadata: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.sequence, self.camera, self.capture_id, self.frame_id

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "A2D2FrameRecord":
        expected_fields = {
            "sequence",
            "camera",
            "capture_id",
            "frame_id",
            "image",
            "label",
            "lidar",
            "metadata",
        }
        if set(values) != expected_fields:
            missing = sorted(expected_fields - set(values))
            extra = sorted(set(values) - expected_fields)
            raise ValueError(
                f"Invalid frame record fields; missing={missing}, extra={extra}"
            )
        if not all(isinstance(values[field], str) for field in expected_fields):
            raise ValueError("Every frame record value must be a string")
        if not all(values[field] for field in expected_fields):
            raise ValueError("Frame record values cannot be empty")
        return cls(**values)
