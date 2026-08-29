# Semantic Costmap Perception Pipeline

This project trains an RGB semantic segmentation model, attaches its predictions
to calibrated LiDAR points, and converts those positioned semantics into a 2D
navigation costmap. Later stages will accumulate observations with robot poses
and publish the result through ROS 2 for Nav2 and RViz2.

## System flow

```text
RGB frame -> U-Net semantic mask ----+
                                      +-> labeled 3D points -> local costmap
LiDAR + camera calibration -----------+                         |
                                                                v
pose / SLAM ------------------------------------------> persistent costmap
                                                                |
                                                                v
                                                        ROS 2 / Nav2 / RViz2
```

The camera model provides semantic meaning. LiDAR provides metric position and
depth. Calibration relates the two sensors. Pose estimates allow observations
from different frames to occupy a consistent map frame.

## Current class design

The network predicts six classes: drivable, caution, non-drivable, static
obstacle, dynamic obstacle, and background. Background includes sky and receives
no cost. Invalid pixels use ignore ID 255 and are excluded from training.

See `docs/semantic_taxonomy.md` and
`configs/a2d2_semantic_mapping_v2.yaml` for exact definitions.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Training is intended for a GPU Colab runtime. Colab already provides PyTorch;
install this repository there with `python -m pip install -e .`.

## Validate the setup

```bash
python tools/validate_a2d2_mapping.py
pytest
```

## Data preparation commands

Convert downloaded source labels:

```bash
python tools/convert_a2d2_labels.py \
  --input-dir data/raw/a2d2/SEQUENCE/label/cam_front_center \
  --output-dir data/processed/a2d2_v2/SEQUENCE/masks
```

Pair images and converted masks in a manifest:

```bash
python tools/build_a2d2_manifest.py \
  --image-dir data/raw/a2d2/SEQUENCE/camera/cam_front_center \
  --mask-dir data/processed/a2d2_v2/SEQUENCE/masks \
  --sequence SEQUENCE \
  --output data/splits/SEQUENCE.jsonl
```

The next implementation stage is the clean two-notebook workflow documented in
`notebooks/README.md`. Model checkpoints and large datasets are ignored by Git.
