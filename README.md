# Semantic Costmap Perception Pipeline

This project combines RGB semantic segmentation with LiDAR geometry to create
navigation costs for ROS 2 and Nav2.

```text
RGB image -> U-Net -> per-pixel class probabilities
                                |
LiDAR + camera calibration -----+-> painted 3D points
                                      |
                                      v
                              bird's-eye cost grid
                                      |
                                      v
                              ROS 2 / Nav2 costmap
```

The camera predicts **what** is visible. LiDAR measures **where** it is. An
existing localization or SLAM system supplies poses when observations need to
be accumulated in a persistent map; this project does not reimplement SLAM.

## Semantic classes

The trained model predicts five classes:

| ID | Class | Navigation cost |
|---:|---|---:|
| 0 | `drivable` | 0 |
| 1 | `non_drivable` | 220 |
| 2 | `static_obstacle` | 254 |
| 3 | `dynamic_obstacle` | 254 |
| 4 | `background` | not inserted |

`background` includes sky and other image context that has no physical
costmap location.

## Model result

The U-Net was trained from scratch on A2D2. The selected epoch-29 checkpoint
reached 0.8456 navigation mIoU on validation and 0.7966 on the held-out test
split. Checkpoints are local artifacts and are intentionally ignored by Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Place the trained checkpoint at:

```text
outputs/checkpoints/epoch29_restore/best_semantic_unet.pt
```

Run one-image inference with:

```bash
python tools/run_inference.py --device cpu
```

## Repository layout

```text
configs/                 Small configuration and calibration files
docs/                    Architecture and usage documentation
notebooks/               Training notebook notes
src/semantic_costmap/    Reusable offline Python pipeline
tools/                   Runnable command-line demonstrations
tests/                   Unit and integration tests
data/                    Local datasets (ignored)
outputs/                 Generated results and checkpoints (ignored)
```

The remaining pipeline is implemented in this order: independent calibration
projection, semantic point painting, costmap generation, multi-frame playback,
ROS 2/Nav2 integration, pose-aware accumulation, packaging, and documentation.
