import json
from pathlib import Path

from offroad_vision.data.taxonomy import SemanticTaxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "configs/a2d2_semantic_mapping_v2.yaml"
CLASS_LIST_PATH = PROJECT_ROOT / "configs/a2d2_class_list.json"


def test_taxonomy_covers_every_a2d2_class() -> None:
    taxonomy = SemanticTaxonomy.from_yaml(MAPPING_PATH)
    source_classes = json.loads(CLASS_LIST_PATH.read_text()).values()
    taxonomy.validate(source_classes)

    assert taxonomy.num_classes == 6
    assert taxonomy.navigation_ids == (0, 1, 2, 3, 4)
    assert taxonomy.background_ids == frozenset({5})


def test_navigation_meanings_are_explicit() -> None:
    taxonomy = SemanticTaxonomy.from_yaml(MAPPING_PATH)

    assert taxonomy.train_id_for("RD normal street") == 0
    assert taxonomy.train_id_for("Speed bumper") == 1
    assert taxonomy.train_id_for("Sidewalk") == 2
    assert taxonomy.train_id_for("Buildings") == 3
    assert taxonomy.train_id_for("Car 1") == 4
    assert taxonomy.train_id_for("Sky") == 5
    assert taxonomy.train_id_for("Ego car") == 255
    assert taxonomy.train_id_to_cost == {
        0: 0,
        1: 80,
        2: 220,
        3: 254,
        4: 254,
        5: None,
    }
