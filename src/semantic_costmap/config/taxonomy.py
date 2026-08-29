"""Load and validate the semantic model and navigation class contract."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TaxonomyError(ValueError):
    """Raised when the semantic taxonomy violates the project contract."""


@dataclass(frozen=True)
class SemanticClass:
    """One model output class and its navigation behavior."""

    name: str
    train_id: int
    visualization_rgb: tuple[int, int, int]
    costmap_cost: int | None
    temporal_policy: str
    source_labels: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SemanticClass":
        try:
            return cls(
                name=str(values["name"]),
                train_id=int(values["train_id"]),
                visualization_rgb=tuple(values["visualization_rgb"]),
                costmap_cost=(
                    None
                    if values.get("costmap_cost") is None
                    else int(values["costmap_cost"])
                ),
                temporal_policy=str(values["temporal_policy"]),
                source_labels=tuple(str(label) for label in values["source_labels"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TaxonomyError(f"Invalid semantic class entry: {values!r}") from error


@dataclass(frozen=True)
class SemanticTaxonomy:
    """Validated semantic IDs shared by training, fusion, and costmaps."""

    version: int
    dataset: str
    ignore_id: int
    declared_num_model_classes: int
    classes: tuple[SemanticClass, ...]
    ignored_source_labels: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticTaxonomy":
        path = Path(path)
        try:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise TaxonomyError(f"Could not load taxonomy from {path}: {error}") from error

        if not isinstance(values, Mapping):
            raise TaxonomyError("Taxonomy YAML must contain a top-level mapping")

        raw_classes = values.get("classes")
        if not isinstance(raw_classes, list):
            raise TaxonomyError("Taxonomy 'classes' must be a list")

        try:
            taxonomy = cls(
                version=int(values["version"]),
                dataset=str(values["dataset"]),
                ignore_id=int(values["ignore_id"]),
                declared_num_model_classes=int(values["num_model_classes"]),
                classes=tuple(SemanticClass.from_mapping(item) for item in raw_classes),
                ignored_source_labels=tuple(
                    str(label) for label in values.get("ignored_source_labels", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TaxonomyError("Taxonomy is missing required fields") from error

        taxonomy.validate()
        return taxonomy

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_by_id(self) -> dict[int, SemanticClass]:
        return {semantic_class.train_id: semantic_class for semantic_class in self.classes}

    @property
    def class_by_name(self) -> dict[str, SemanticClass]:
        return {semantic_class.name: semantic_class for semantic_class in self.classes}

    @property
    def source_label_to_train_id(self) -> dict[str, int]:
        return {
            source_label: semantic_class.train_id
            for semantic_class in self.classes
            for source_label in semantic_class.source_labels
        }

    @property
    def navigation_train_ids(self) -> tuple[int, ...]:
        return tuple(
            semantic_class.train_id
            for semantic_class in self.classes
            if semantic_class.costmap_cost is not None
        )

    def train_id_for_source_label(self, source_label: str) -> int:
        if source_label in self.ignored_source_labels:
            return self.ignore_id
        try:
            return self.source_label_to_train_id[source_label]
        except KeyError as error:
            raise TaxonomyError(f"Unmapped source label: {source_label!r}") from error

    def cost_for_train_id(self, train_id: int) -> int | None:
        try:
            return self.class_by_id[train_id].costmap_cost
        except KeyError as error:
            raise TaxonomyError(f"Unknown training ID: {train_id}") from error

    def validate(self, expected_source_labels: Iterable[str] | None = None) -> None:
        if self.version < 1:
            raise TaxonomyError("Taxonomy version must be positive")
        if not self.dataset.strip():
            raise TaxonomyError("Dataset name cannot be empty")
        if not 0 <= self.ignore_id <= 255:
            raise TaxonomyError("Ignore ID must fit in an unsigned 8-bit mask")
        if self.declared_num_model_classes != self.num_classes:
            raise TaxonomyError(
                "num_model_classes does not match the number of class entries"
            )

        train_ids = [semantic_class.train_id for semantic_class in self.classes]
        expected_ids = list(range(self.num_classes))
        if sorted(train_ids) != expected_ids:
            raise TaxonomyError(
                f"Training IDs must be contiguous {expected_ids}; got {sorted(train_ids)}"
            )
        if self.ignore_id in train_ids:
            raise TaxonomyError("Ignore ID cannot also be a model output ID")

        names = [semantic_class.name for semantic_class in self.classes]
        duplicate_names = _duplicates(names)
        if duplicate_names:
            raise TaxonomyError(f"Duplicate class names: {duplicate_names}")

        colors: list[tuple[int, int, int]] = []
        learned_labels: list[str] = []
        for semantic_class in self.classes:
            if not semantic_class.name.strip():
                raise TaxonomyError("Class names cannot be empty")
            if not semantic_class.temporal_policy.strip():
                raise TaxonomyError(
                    f"Temporal policy is empty for {semantic_class.name}"
                )
            if not semantic_class.source_labels:
                raise TaxonomyError(
                    f"Class has no source labels: {semantic_class.name}"
                )

            color = semantic_class.visualization_rgb
            if (
                len(color) != 3
                or any(isinstance(channel, bool) for channel in color)
                or any(not isinstance(channel, int) for channel in color)
                or any(not 0 <= channel <= 255 for channel in color)
            ):
                raise TaxonomyError(
                    f"Invalid visualization color for {semantic_class.name}: {color}"
                )
            colors.append(color)

            cost = semantic_class.costmap_cost
            if cost is not None and not 0 <= cost <= 254:
                raise TaxonomyError(
                    f"Invalid Nav2 cost for {semantic_class.name}: {cost}"
                )
            learned_labels.extend(semantic_class.source_labels)

        duplicate_colors = _duplicates(colors)
        if duplicate_colors:
            raise TaxonomyError(f"Duplicate visualization colors: {duplicate_colors}")

        duplicate_learned = _duplicates(learned_labels)
        if duplicate_learned:
            raise TaxonomyError(
                f"Source labels assigned to multiple classes: {duplicate_learned}"
            )

        duplicate_ignored = _duplicates(self.ignored_source_labels)
        if duplicate_ignored:
            raise TaxonomyError(f"Duplicate ignored labels: {duplicate_ignored}")

        overlap = set(learned_labels) & set(self.ignored_source_labels)
        if overlap:
            raise TaxonomyError(
                f"Labels cannot be both learned and ignored: {sorted(overlap)}"
            )

        if expected_source_labels is not None:
            expected = set(expected_source_labels)
            configured = set(learned_labels) | set(self.ignored_source_labels)
            missing = sorted(expected - configured)
            extra = sorted(configured - expected)
            if missing or extra:
                raise TaxonomyError(
                    f"Source-label coverage mismatch; missing={missing}, extra={extra}"
                )


def load_source_class_names(path: str | Path) -> tuple[str, ...]:
    path = Path(path)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TaxonomyError(f"Could not load source class list from {path}: {error}") from error

    if not isinstance(values, dict) or not all(
        isinstance(color, str) and isinstance(name, str)
        for color, name in values.items()
    ):
        raise TaxonomyError("Source class list must map color strings to class names")
    if duplicates := _duplicates(values.values()):
        raise TaxonomyError(f"Duplicate source class names: {duplicates}")
    return tuple(values.values())


def _duplicates(values: Iterable[Any]) -> list[Any]:
    return sorted(value for value, count in Counter(values).items() if count > 1)
