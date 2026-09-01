from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from offroad_vision.models.semantic_unet import SemanticUNet


IMAGE_PATH = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)

LIDAR_PATH = Path(
    "data/raw/a2d2_sample/lidar/"
    "20180807145028_lidar_frontcenter_000000091.npz"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/epoch29_restore/"
    "best_semantic_unet.pt"
)

OUTPUT_DIR = Path(
    "outputs/fusion"
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

CLASS_COSTS = np.array(
    [0, 220, 254, 254, -1],
    dtype=np.int16,
)

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

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    original_width, original_height = image.size

    resized_image = image.resize(
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
        small_prediction = logits.argmax(
            dim=1
        )[0].cpu().numpy()

    prediction = np.asarray(
        Image.fromarray(
            small_prediction.astype(np.uint8),
            mode="L",
        ).resize(
            (original_width, original_height),
            Image.Resampling.NEAREST,
        )
    )

    lidar_data = np.load(LIDAR_PATH)

    rows = np.rint(
        lidar_data["row"]
    ).astype(np.int32)

    cols = np.rint(
        lidar_data["col"]
    ).astype(np.int32)

    valid = (
        (rows >= 0)
        & (rows < original_height)
        & (cols >= 0)
        & (cols < original_width)
    )

    rows = rows[valid]
    cols = cols[valid]

    points = lidar_data["points"][valid]
    depths = lidar_data["depth"][valid]
    lidar_ids = lidar_data["lidar_id"][valid]

    predicted_classes = prediction[
        rows,
        cols,
    ]

    predicted_costs = CLASS_COSTS[
        predicted_classes
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    semantic_points_path = (
        OUTPUT_DIR / "semantic_lidar_points.npz"
    )

    np.savez_compressed(
        semantic_points_path,
        points=points,
        row=rows,
        col=cols,
        depth=depths,
        lidar_id=lidar_ids,
        predicted_class=predicted_classes,
        predicted_cost=predicted_costs,
    )

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for row, col, class_id in zip(
        rows,
        cols,
        predicted_classes,
    ):
        color = tuple(
            int(value)
            for value in CLASS_COLORS[class_id]
        )

        draw.ellipse(
            (
                int(col) - 2,
                int(row) - 2,
                int(col) + 2,
                int(row) + 2,
            ),
            fill=color,
        )

    overlay_path = (
        OUTPUT_DIR / "semantic_lidar_overlay.png"
    )

    overlay.save(overlay_path)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Valid LiDAR points:", len(points))
    print("Saved:", semantic_points_path)
    print("Saved:", overlay_path)

    print("\nPredicted point classes:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        count = np.count_nonzero(
            predicted_classes == class_id
        )

        print(
            f"{class_name}: {count}"
        )


if __name__ == "__main__":
    main()