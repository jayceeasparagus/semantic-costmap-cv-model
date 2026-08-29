import pytest

torch = pytest.importorskip("torch")

from offroad_vision.training.metrics import confusion_matrix, segmentation_summary


def test_metrics_ignore_255_and_report_navigation_miou() -> None:
    predictions = torch.tensor([[0, 1, 1], [0, 1, 0]])
    targets = torch.tensor([[0, 1, 0], [0, 1, 255]])

    matrix = confusion_matrix(predictions, targets, num_classes=2, ignore_id=255)
    summary = segmentation_summary(
        matrix,
        class_names={0: "drivable", 1: "background"},
        navigation_ids=(0,),
    )

    assert matrix.tolist() == [[2, 1], [0, 2]]
    assert summary["pixel_accuracy"] == pytest.approx(0.8)
    assert summary["navigation_miou"] == pytest.approx(2 / 3)
