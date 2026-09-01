from pathlib import Path

import numpy as np
import torch
from PIL import Image

from offroad_vision.models.semantic_unet import SemanticUNet


IMAGE_PATH = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/epoch29_restore/"
    "best_semantic_unet.pt"
)

OUTPUT_DIR = Path(
    "outputs/inference"
)

IMAGE_SIZE = (320, 512)

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

CLASS_NAMES = [
    "drivable",
    "non_drivable",
    "static_obstacle",
    "dynamic_obstacle",
    "background",
]

CLASS_COLORS = np.array(
    [
        [0, 200, 0],
        [255, 128, 0],
        [255, 0, 0],
        [255, 0, 255],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def main():
    device = torch.device("cpu")

    model = SemanticUNet(
        num_classes=5,
        base_channels=32,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    original_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    resized_image = original_image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    ) / 255.0

    image_array = (
        image_array - IMAGE_MEAN
    ) / IMAGE_STD

    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        prediction = logits.argmax(
            dim=1
        )[0].cpu().numpy()

        confidence = probabilities.max(
            dim=1
        )[0][0].cpu().numpy()

    predicted_mask = Image.fromarray(
        prediction.astype(np.uint8),
        mode="L",
    ).resize(
        original_image.size,
        Image.Resampling.NEAREST,
    )

    small_color_mask = Image.fromarray(
        CLASS_COLORS[prediction],
        mode="RGB",
    )

    color_mask = small_color_mask.resize(
        original_image.size,
        Image.Resampling.NEAREST,
    )

    overlay = Image.blend(
        original_image,
        color_mask,
        alpha=0.45,
    )

    confidence_image = Image.fromarray(
        (confidence * 255).astype(np.uint8),
        mode="L",
    ).resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_mask.save(
        OUTPUT_DIR / "sample_prediction_ids.png"
    )

    color_mask.save(
        OUTPUT_DIR / "sample_prediction_colors.png"
    )

    overlay.save(
        OUTPUT_DIR / "sample_prediction_overlay.png"
    )

    confidence_image.save(
        OUTPUT_DIR / "sample_prediction_confidence.png"
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Saved outputs to:", OUTPUT_DIR)

    for class_id, class_name in enumerate(CLASS_NAMES):
        percentage = (
            np.mean(prediction == class_id)
            * 100
        )

        print(
            f"{class_name}: "
            f"{percentage:.2f}%"
        )


if __name__ == "__main__":
    main()