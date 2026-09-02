from semantic_costmap.config import (
    BACKGROUND_CLASS_ID,
    NUM_CLASSES,
    SEMANTIC_CLASSES,
    class_costs,
)


def test_semantic_classes_have_contiguous_ids() -> None:
    assert [item.class_id for item in SEMANTIC_CLASSES] == list(range(NUM_CLASSES))


def test_background_has_no_navigation_cost() -> None:
    assert SEMANTIC_CLASSES[BACKGROUND_CLASS_ID].cost is None
    assert class_costs(background_value=-1)[BACKGROUND_CLASS_ID] == -1
