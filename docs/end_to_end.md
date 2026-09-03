# End-to-end runbook

This project has two complementary demonstrations:

1. the offline A2D2 pipeline, which runs the trained U-Net, calibrates LiDAR
   points, paints them with semantic predictions, and rasterizes a local
   metric costmap;
2. the ROS 2 integration, which replays A2D2 sensors, supplies odometry,
   lets SLAM Toolbox publish `map -> odom`, and exposes the semantic grid to
   Nav2.

## Offline pipeline

From the repository root, activate the virtual environment and run:

```bash
source .venv/bin/activate
export PYTHONPATH=src

python tools/run_inference.py --device cpu
python tools/validate_calibration_projection.py
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
python tools/run_playback.py --device cpu --max-frames 8
```

The commands write diagnostic images and arrays under `outputs/`. The
checkpoint is intentionally ignored by Git and must be restored locally at
`outputs/checkpoints/epoch29_restore/best_semantic_unet.pt`.

For pose-aware accumulation, first build the pose CSV from the A2D2 bus
signals, then pass it to playback:

```bash
python tools/build_a2d2_poses.py
python tools/run_playback.py --device cpu --max-frames 8 \
  --poses-csv outputs/poses/20180807_bus_odometry.csv
```

## ROS 2 replay and SLAM

The ROS packages target Jazzy. After installing ROS dependencies and building
the workspace:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
export PYTHONPATH="$PWD/ros2/semantic_costmap_ros:$PYTHONPATH"
ros2 launch semantic_costmap_ros a2d2_slam.launch.py
```

The replay node publishes the camera, calibrated point cloud, and bus-derived
`odom -> base_link` transform. `pointcloud_to_laserscan` converts the cloud to
the scan interface expected by SLAM Toolbox. SLAM Toolbox owns the
`map -> odom` transform, while the semantic node and accumulator publish the
local and persistent semantic grids.

This replay is deterministic and finite by default. It requires local A2D2
camera, LiDAR, calibration, and bus files; those files are never committed.

To attach the Nav2 planner and custom global semantic layer to the same graph,
use:

```bash
ros2 launch semantic_costmap_ros a2d2_slam_nav2.launch.py
```

The replay uses current ROS timestamps even though its pose increments come
from recorded bus signals, because Nav2 rejects stale sensor layers. The
integrated verifier writes `outputs/slam_nav2/integration_result.json` only
after observing the SLAM map, `map -> odom`, the composed
`map -> base_link` pose, non-empty persistent semantic evidence, and Nav2's
global costmap.

## Headless Nav2 planning proof

The Nav2 demo is independent of the large A2D2 download. It uses a small
deterministic map and a switchable semantic barrier so the planner behavior is
easy to test:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export PYTHONPATH="$PWD/ros2/semantic_costmap_ros:$PYTHONPATH"
ros2 launch semantic_costmap_ros nav2_demo.launch.py
```

The client sends `ComputePathToPose` twice. It verifies that both paths reach
the goal, the semantic-aware path avoids lethal cells, and enabling the
semantic layer changes the route. The evidence is saved under
`outputs/nav2_demo/`:

- `planning_result.json` contains path lengths and route-change data;
- `path_without_semantics.json` and `path_with_semantics.json` contain the
  returned paths;
- `path_overlay.png` shows the red semantic barrier, blue baseline route,
  green semantic-aware route, and start/goal markers.

This is a headless integration test of the costmap-to-planner connection. A
robot deployment would replace the deterministic publisher with the semantic
costmap node and use Nav2's normal controller and behavior-tree stack.

## What is and is not demonstrated

- Camera semantics identify the class of image pixels; LiDAR supplies metric
  depth and 3D obstacle evidence.
- Calibration maps LiDAR returns into the camera image, and the painted labels
  are rasterized into a vehicle-frame grid.
- Ray tracing marks observed free space, obstacle footprints improve sparse
  returns, and the Nav2 inflation layer expands collision cost around lethal
  cells.
- The integrated launch demonstrates bus-derived odometry corrected through
  SLAM Toolbox and consumed by semantic accumulation and Nav2, but it is not a
  claim of localization accuracy.
- The measured local CPU playback rate and model metrics in
  `docs/benchmark_results.md` are the project’s reported performance numbers;
  no real-time claim is made for an arbitrary robot computer.
