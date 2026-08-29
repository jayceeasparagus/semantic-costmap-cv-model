# Semantic Costmap Perception Pipeline

This project will combine camera semantics with LiDAR geometry to produce a
navigation costmap for ROS 2 and Nav2.

## What the system will do

```text
RGB frame -> segmentation model -> per-pixel semantic probabilities
                                           |
LiDAR points + camera calibration ----------+
                                           v
                              semantically labeled 3D points
                                           |
                                           v
                                 bird's-eye cost grid
                                           |
                                           v
                                      ROS 2 / Nav2
```

The camera model answers **what a pixel represents**. LiDAR answers **where that
observation is in 3D**. The costmap converts the fused result into navigation
meaning such as free, caution, strongly avoided, or lethal.

SLAM is not part of the model. Later, an existing SLAM or localization system
will provide poses so observations can be placed consistently in the map frame.

## Initial model classes

The first model will predict six classes:

1. `drivable`
2. `caution`
3. `non_drivable`
4. `static_obstacle`
5. `dynamic_obstacle`
6. `background`

`background` includes sky and other visible context that must not enter the
costmap. These classes are provisional until the A2D2 data audit is complete.

## Repository layout

```text
configs/       Dataset metadata that the code reads
docs/          Architecture and project explanations
notebooks/     Colab training and experiments
data/          Local datasets; ignored by Git
outputs/       Checkpoints, figures, and generated maps; ignored by Git
```

Folders for reusable Python code, ROS 2, tests, and Docker will be added when
those parts begin. This keeps the repository easy to understand while avoiding
empty or premature infrastructure.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch is intentionally not pinned yet. Training will run in Colab, whose
PyTorch build must match its GPU runtime. We will record the working training
environment when the model notebook is stable.

## Development order

1. Audit A2D2 labels and build a clean training dataset.
2. Train and evaluate the RGB segmentation model.
3. Project synchronized LiDAR points into the camera image.
4. Attach semantic probabilities to the projected points.
5. Rasterize those points into a bird's-eye semantic cost grid.
6. Add temporal placement using supplied poses.
7. Integrate the grid with ROS 2 and a Nav2 costmap layer.
8. Add tests, benchmarks, Docker, and final documentation.

Current status: repository simplification complete; implementation is paused.
