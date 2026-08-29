"""Semantic class taxonomy shared by preprocessing, training, and costmaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class SemanticGroup:
    """One model output class and its eventual navigation cost."""

    name: str
    train_id: int
    costmap_cost: int | None
    classes: tuple[str, ...]


@dataclass(frozen=True)
class SemanticTaxonomy:
    """Validated mapping from source labels to compact training IDs."""

    version: int
    ignore_id: int
    groups: tuple[SemanticGroup, ...]
    ignored_classes: frozenset[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticTaxonomy":
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Taxonomy YAML must contain a mapping")

        raw_groups = config.get("groups")
        if not isinstance(raw_groups, dict) or not raw_groups:
            raise ValueError("Taxonomy must define at least one group")

        groups = tuple(
            SemanticGroup(
                name=name,
                train_id=int(values["train_id"]),
                costmap_cost=(
                    None
                    if values.get("costmap_cost") is None
                    else int(values["costmap_cost"])
                ),
                classes=tuple(values.get("classes", [])),
            )
            for name, values in raw_groups.items()
        )

        taxonomy = cls(
            version=int(config.get("version", 1)),
            ignore_id=int(config.get("ignore_id", 255)),
            groups=groups,
            ignored_classes=frozenset(config.get("ignore", [])),
        )
        taxonomy.validate()
        return taxonomy

    @property
    def num_classes(self) -> int:
        return len(self.groups)

    @property
    def class_to_train_id(self) -> dict[str, int]:
        return {
            class_name: group.train_id
            for group in self.groups
            for class_name in group.classes
        }

    @property
    def train_id_to_name(self) -> dict[int, str]:
        return {group.train_id: group.name for group in self.groups}

    @property
    def train_id_to_cost(self) -> dict[int, int | None]:
        return {group.train_id: group.costmap_cost for group in self.groups}

    @property
    def background_ids(self) -> frozenset[int]:
        return frozenset(
            group.train_id
            for group in self.groups
            if group.costmap_cost is None
        )

    @property
    def navigation_ids(self) -> tuple[int, ...]:
        return tuple(
            group.train_id
            for group in self.groups
            if group.costmap_cost is not None
        )

    def train_id_for(self, source_class: str) -> int:
        if source_class in self.ignored_classes:
            return self.ignore_id
        try:
            return self.class_to_train_id[source_class]
        except KeyError as error:
            raise KeyError(f"Unmapped source class: {source_class}") from error

    def validate(self, source_classes: Iterable[str] | None = None) -> None:
        if not 0 <= self.ignore_id <= 255:
            raise ValueError("ignore_id must fit in an unsigned 8-bit mask")

        train_ids = [group.train_id for group in self.groups]
        expected_ids = list(range(len(self.groups)))
        if sorted(train_ids) != expected_ids:
            raise ValueError(
                f"Training IDs must be contiguous {expected_ids}; got {sorted(train_ids)}"
            )
        if self.ignore_id in train_ids:
            raise ValueError("ignore_id cannot also be a training class")

        assigned_classes: list[str] = []
        for group in self.groups:
            if not group.classes:
                raise ValueError(f"Group has no source classes: {group.name}")
            if group.costmap_cost is not None and not 0 <= group.costmap_cost <= 254:
                raise ValueError(
                    f"Invalid costmap cost for {group.name}: {group.costmap_cost}"
                )
            assigned_classes.extend(group.classes)

        duplicates = sorted(
            class_name
            for class_name in set(assigned_classes)
            if assigned_classes.count(class_name) > 1
        )
        if duplicates:
            raise ValueError(f"Classes assigned more than once: {duplicates}")

        overlap = set(assigned_classes) & self.ignored_classes
        if overlap:
            raise ValueError(f"Classes cannot be both learned and ignored: {sorted(overlap)}")

        if source_classes is not None:
            expected = set(source_classes)
            configured = set(assigned_classes) | set(self.ignored_classes)
            missing = sorted(expected - configured)
            extra = sorted(configured - expected)
            if missing or extra:
                raise ValueError(
                    f"Taxonomy/source mismatch; missing={missing}, extra={extra}"
                )
