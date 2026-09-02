import torch

from semantic_costmap.models import SemanticUNet


def test_unet_output_shape_and_parameter_count() -> None:
    model = SemanticUNet(num_classes=5, base_channels=32)
    inputs = torch.zeros((1, 3, 64, 96), dtype=torch.float32)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (1, 5, 64, 96)
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_762_693
