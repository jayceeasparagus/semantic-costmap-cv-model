# Semantic Costmap Perception Pipeline

An end-to-end perception project that combines camera semantics with LiDAR
geometry and exposes the result as ROS 2 and Nav2 costmaps.

```text
RGB frame -> U-Net -> per-pixel semantic probabilities
                                  |
LiDAR + camera calibration -------+-> semantically painted 3D points
                                         |
                                         v
                                vehicle-relative cost grid
                                         |
                         +---------------+----------------+
                         |                                |
                    Nav2 layer                 SLAM pose accumulation
```

The camera predicts **what** is present, LiDAR measures **where** it is, and an
existing localization or SLAM system supplies poses for persistent mapping.
The project deliberately uses Nav2 and SLAM Toolbox as infrastructure instead
of recreating navigation and localization.

## Results

The U-Net was trained from scratch on A2D2 with five navigation-oriented
classes. The selected epoch-29 checkpoint achieved:

- **0.8456** navigation mIoU on validation;
- **0.7966** navigation mIoU and **0.8314** all-class mIoU on the test split;
- **0.9439 / 0.8707 / 0.6452 / 0.7265** test IoU for drivable,
  non-drivable, static-obstacle, and dynamic-obstacle classes.

An eight-frame dense-costmap CPU playback ran at **0.60 FPS**. U-Net inference
averaged 1535.4 ms, while projection, semantic fusion, and costmap generation
together averaged 30.4 ms. See [benchmark results](docs/benchmark_results.md).

## Navigation classes

| ID | Class | Nav2 cost | Meaning |
|---:|---|---:|---|
| 0 | `drivable` | 0 | preferred ground |
| 1 | `non_drivable` | 220 | strongly avoid |
| 2 | `static_obstacle` | 254 | fixed collision hazard |
| 3 | `dynamic_obstacle` | 254 | moving collision hazard |
| 4 | `background` | skipped | sky and non-spatial context |

Raw LiDAR obstacle evidence can raise a cell's cost but semantic predictions
cannot lower it. Cells outside observed rays remain unknown.

LiDAR ray tracing marks observed space before each return as free, and a
conservative neighbor rule fills small gaps surrounded by drivable evidence.
On the included sample these steps increased single-frame known coverage from
4.65% to 45.40% without lowering obstacle costs.

## Local setup

ROS-independent tools require Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Place the trained checkpoint at:

```text
outputs/checkpoints/epoch29_restore/best_semantic_unet.pt
```

The default demos also expect paired A2D2 front-center samples under
`data/raw/a2d2_sample/` and playback frames under
`data/raw/a2d2_playback/`. Data, checkpoints, and generated outputs are ignored
by Git.

## Offline demonstration

Run each stage from the repository root:

```bash
# 1. Segment one RGB image.
python tools/run_inference.py --device cpu

# 2. Recompute LiDAR image coordinates from calibration.
python tools/validate_calibration_projection.py

# 3. Attach semantic probabilities and costs to 3D points.
python tools/paint_semantic_points.py --device cpu

# 4. Rasterize the painted cloud into a local metric grid.
python tools/generate_costmap.py

# 5. Process a synchronized frame sequence and measure latency.
python tools/run_playback.py --device cpu --max-frames 8

# 6. Place local grids into a persistent map using example poses.
python tools/demo_pose_accumulation.py
```

For pose-aligned sequence accumulation, provide a CSV containing
`frame_id,timestamp,x,y,yaw` and add `--poses-csv path/to/poses.csv` to the
playback command. These are map-to-base poses from odometry, localization, or
SLAM.

Each tool writes inspectable images and arrays under `outputs/`. Detailed data
flow and equations are in [the architecture guide](docs/architecture.md).

## ROS 2 and Nav2

The ROS packages target ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
ros2 launch semantic_costmap_ros semantic_costmap.launch.py
```

`semantic_costmap_node` consumes synchronized image, camera-info, point-cloud,
and TF data. It publishes a semantic mask, painted cloud, and local
`OccupancyGrid`. `semantic_map_accumulator` uses the standard
`map -> odom -> base_link` TF chain to publish a persistent semantic grid.

The C++ `semantic_costmap_layer` plugin max-merges either grid into Nav2 before
inflation. Configuration examples are in the [ROS 2 node](docs/ros2_node.md),
[Nav2 layer](docs/nav2_layer.md), and [SLAM accumulation](docs/slam_accumulation.md)
guides.

`ros2/semantic_costmap_ros/config/nav2_semantic_layers.yaml` configures Nav2's
standard inflation layer after the semantic layer for both local and global
costmaps.

The deterministic headless Nav2 planning proof can be run with:

```bash
export PYTHONPATH="$PWD/ros2/semantic_costmap_ros:$PYTHONPATH"
ros2 launch semantic_costmap_ros nav2_demo.launch.py
```

It saves both returned paths and a route overlay under `outputs/nav2_demo/`.
See the [end-to-end runbook](docs/end_to_end.md) for the complete workflow.

With local A2D2 replay data available, one integrated launch runs replay,
SLAM Toolbox, persistent semantic accumulation, and the Nav2 planner:

```bash
ros2 launch semantic_costmap_ros a2d2_slam_nav2.launch.py
```

SLAM Toolbox publishes `map -> odom`; replay publishes `odom -> base_link`;
and the accumulator transforms each painted cloud through that composed pose.
Nav2's global costmap max-merges `/semantic_global_costmap` through the custom
C++ layer before inflation. A successful live wiring check is written to
`outputs/slam_nav2/integration_result.json`. The launch also selects a distant
low-cost goal cell, requests a Nav2 path, and saves
`outputs/slam_nav2/nav2_path_overlay.png` with route metadata in
`nav2_goal_result.json`.

## Tests

```bash
source .venv/bin/activate
tools/run_checks.sh
```

The complete local check runs 36 Python tests, builds both ROS packages, runs
five ROS/C++ tests, verifies plugin registration, and performs node smoke
tests. The A2D2 integration test skips when local data or the checkpoint is not
available. Portable Python checks also run in GitHub Actions.

## Docker

```bash
docker compose build
docker compose run --rm semantic-costmap
```

The image contains the Python package, CPU PyTorch, ROS 2 Jazzy, Nav2, SLAM
Toolbox, and both local ROS packages. Datasets, outputs, and checkpoints are
mounted at runtime rather than copied into the image. See [Docker usage](docs/docker.md).

## Repository layout

```text
configs/                 A2D2 calibration and label definitions
docs/                    Design, integration, and test documentation
src/semantic_costmap/    Shared inference, geometry, fusion, and mapping code
tools/                   Offline demos and validation commands
ros2/                    ROS 2 nodes and the Nav2 C++ layer
tests/                   Unit and end-to-end integration tests
docker/                  Container entrypoint
data/                    Local datasets (ignored)
outputs/                 Generated artifacts and checkpoints (ignored)
```

## Scope and limitations

This is a tested offline and headless ROS integration, not a vehicle-certified
system. The included A2D2 sample provides camera-frame LiDAR points; the code
still computes image projection independently from calibration. The ROS
integration uses bus odometry as SLAM Toolbox's initial motion estimate and
laser scan matching to provide the map-frame correction. This demonstrates the
software integration, not localization accuracy. GPU inference or model
optimization is needed for practical real-time frame rates.
