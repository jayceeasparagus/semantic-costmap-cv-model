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

An eight-frame CPU playback ran at **0.79 FPS**. U-Net inference averaged
1155.5 ms, while projection, semantic fusion, and costmap generation together
averaged 10.3 ms. See [benchmark results](docs/benchmark_results.md).

## Navigation classes

| ID | Class | Nav2 cost | Meaning |
|---:|---|---:|---|
| 0 | `drivable` | 0 | preferred ground |
| 1 | `non_drivable` | 220 | strongly avoid |
| 2 | `static_obstacle` | 254 | fixed collision hazard |
| 3 | `dynamic_obstacle` | 254 | moving collision hazard |
| 4 | `background` | skipped | sky and non-spatial context |

Raw LiDAR obstacle evidence can raise a cell's cost but semantic predictions
cannot lower it. Unknown cells remain unknown.

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

## Tests

```bash
source .venv/bin/activate
tools/run_checks.sh
```

The complete local check runs 22 Python tests, builds both ROS packages, runs
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
still computes image projection independently from calibration. The persistent
map demo uses synthetic poses because the small sample has no ROS TF stream.
A live robot or rosbag must provide synchronized sensors and valid TF from a
real SLAM/localization system. GPU inference or model optimization is needed
for practical real-time frame rates.
