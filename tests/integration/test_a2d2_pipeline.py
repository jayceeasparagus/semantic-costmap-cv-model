from pathlib import Path

import numpy as np
import pytest

from semantic_costmap.config import DEFAULT_CHECKPOINT_PATH
from semantic_costmap.pipeline import SemanticCostmapPipeline


IMAGE_PATH = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)
LIDAR_PATH = Path(
    "data/raw/a2d2_sample/lidar/"
    "20180807145028_lidar_frontcenter_000000091.npz"
)
CALIBRATION_PATH = Path("configs/a2d2_cams_lidars.json")


@pytest.mark.integration
def test_epoch29_sample_runs_end_to_end():
    required = [
        DEFAULT_CHECKPOINT_PATH,
        IMAGE_PATH,
        LIDAR_PATH,
        CALIBRATION_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("local model/data artifacts are unavailable: " + ", ".join(missing))

    pipeline = SemanticCostmapPipeline(
        DEFAULT_CHECKPOINT_PATH,
        CALIBRATION_PATH,
        device="cpu",
    )
    result = pipeline.process(IMAGE_PATH, LIDAR_PATH)

    assert pipeline.segmenter.epoch == 29
    assert result.segmentation.class_ids.shape == (1208, 1920)
    assert result.painted_points.probabilities.shape == (9261, 5)
    assert np.isfinite(result.painted_points.probabilities).all()
    assert result.costmap.costs.shape == (200, 250)
    assert result.costmap.obstacle_mask.any()
    assert result.timings_ms["total"] > 0.0
