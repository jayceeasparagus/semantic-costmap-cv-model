import pytest

torch = pytest.importorskip("torch")

from offroad_vision.models import SemanticSegmentationUNet


def test_unet_preserves_spatial_shape() -> None:
    model = SemanticSegmentationUNet(num_classes=6, base_channels=8)
    inputs = torch.randn(2, 3, 65, 97)

    logits = model(inputs)

    assert logits.shape == (2, 6, 65, 97)
