from pathlib import Path

import pytest

from semantic_costmap.config import (
    SemanticTaxonomy,
    TaxonomyError,
    load_source_class_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = PROJECT_ROOT / "configs/semantic_classes.yaml"
SOURCE_CLASSES_PATH = PROJECT_ROOT / "configs/a2d2_class_list.json"


@pytest.fixture(scope="module")
def taxonomy() -> SemanticTaxonomy:
    return SemanticTaxonomy.from_yaml(TAXONOMY_PATH)


def test_taxonomy_covers_every_a2d2_label_once(
    taxonomy: SemanticTaxonomy,
) -> None:
    source_labels = load_source_class_names(SOURCE_CLASSES_PATH)

    taxonomy.validate(source_labels)

    learned_count = sum(
        len(semantic_class.source_labels) for semantic_class in taxonomy.classes
    )
    assert len(source_labels) == 55
    assert learned_count == 52
    assert len(taxonomy.ignored_source_labels) == 3


def test_navigation_contract(taxonomy: SemanticTaxonomy) -> None:
    assert taxonomy.num_classes == 6
    assert taxonomy.navigation_train_ids == (0, 1, 2, 3, 4)
    assert taxonomy.class_by_id[5].name == "background"

    assert taxonomy.train_id_for_source_label("RD normal street") == 0
    assert taxonomy.train_id_for_source_label("Speed bumper") == 1
    assert taxonomy.train_id_for_source_label("Sidewalk") == 2
    assert taxonomy.train_id_for_source_label("Curbstone") == 3
    assert taxonomy.train_id_for_source_label("Car 1") == 4
    assert taxonomy.train_id_for_source_label("Sky") == 5
    assert taxonomy.train_id_for_source_label("Ego car") == 255

    assert taxonomy.cost_for_train_id(0) == 0
    assert taxonomy.cost_for_train_id(1) == 80
    assert taxonomy.cost_for_train_id(2) == 220
    assert taxonomy.cost_for_train_id(3) == 254
    assert taxonomy.cost_for_train_id(4) == 254
    assert taxonomy.cost_for_train_id(5) is None


def test_unknown_source_label_is_rejected(taxonomy: SemanticTaxonomy) -> None:
    with pytest.raises(TaxonomyError, match="Unmapped source label"):
        taxonomy.train_id_for_source_label("Imaginary road material")


def test_incomplete_source_coverage_is_rejected(
    taxonomy: SemanticTaxonomy,
) -> None:
    source_labels = load_source_class_names(SOURCE_CLASSES_PATH)

    with pytest.raises(TaxonomyError, match="coverage mismatch"):
        taxonomy.validate((*source_labels, "Imaginary source class"))
