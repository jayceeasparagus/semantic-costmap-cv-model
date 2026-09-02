# System architecture

## 1. Purpose

The pipeline converts synchronized RGB and LiDAR observations into a metric
semantic costmap that can be consumed by Nav2. It separates learned perception
from geometry and navigation so each stage can be inspected and tested.

## 2. End-to-end data flow

```text
RGB image ---------------------> U-Net -----> H x W x 5 probabilities
                                                      |
LiDAR points -> TF/extrinsics -> camera frame --------+-> point painting
CameraInfo/calibration --------> pixel projection ----+       |
                                                              v
                                               XYZ + class probabilities
                                                              |
                                             vehicle-frame rasterization
                                                              |
                                                   local semantic costmap
                                                    /                  \
                                   Nav2 max-merge              map-frame pose
                                                                   |
                                                         persistent accumulator
```

## 3. Semantic segmentation

The 7,762,693-parameter U-Net uses four encoder/decoder levels and skip
connections. It was trained from random initialization on A2D2 RGB images and
reduced labels. The network emits five logits per pixel; softmax turns them
into class probabilities.

| ID | Class | Initial cost |
|---:|---|---:|
| 0 | drivable | 0 |
| 1 | non_drivable | 220 |
| 2 | static_obstacle | 254 |
| 3 | dynamic_obstacle | 254 |
| 4 | background | none |

The reduced taxonomy is intentional: source classes with the same navigation
meaning are grouped, while sky becomes background and never enters the map.
The earlier caution class was removed because A2D2 provided too little support
to learn it reliably.

## 4. Camera-LiDAR fusion

The fusion algorithm is semantic point painting. For every LiDAR return, the
system transforms the point into the camera optical frame and projects it with
the camera intrinsic matrix:

```text
p_camera = T_camera_lidar * p_lidar
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

Points behind the camera or outside the image are rejected. The model's five
probabilities at `(u, v)` are sampled and stored with the 3D point. This gives
camera semantics physical depth without asking the RGB network to estimate
distance.

A2D2's `.npz` file includes reference `row` and `col` values, but production
code does not consume them. The calibration validator compares those values
against independently projected coordinates as a correctness test.

## 5. Local costmap generation

Camera-painted points are transformed into the vehicle frame, filtered by
height and confidence, and assigned to grid cells:

```text
cell_x = floor((point_x - origin_x) / resolution)
cell_y = floor((point_y - origin_y) / resolution)
```

Each cell averages its accumulated class probabilities and converts the result
to a probability-weighted navigation cost. Small unknown gaps are filled only
when they have enough neighboring drivable cells. This is local interpolation,
not an assumption that every unknown cell is road.

Each LiDAR return also defines an observed free-space ray from the sensor to the
return. A vectorized grid-ray sampler marks previously unknown cells along that
ray as free but excludes the endpoint. Raw LiDAR endpoints in the configured
obstacle height band are then max-merged as lethal. This ordering prevents
interpolation or free-space clearing from lowering a physical obstacle. The
default grid is 50 m forward by 40 m wide at 0.20 m per cell.

## 6. Multi-frame execution

`SemanticCostmapPipeline` keeps the model and calibration loaded while frames
are processed. `run_playback.py` pairs camera and LiDAR files by frame ID,
generates debug panels and a GIF, and records load, inference, projection,
fusion, costmap, and total latency. When given timestamped map-to-base poses in
a CSV, it also accumulates every local grid into one persistent global map.
Offline file playback models the same per-frame flow that the ROS nodes execute
on live messages and TF poses.

## 7. ROS 2 and Nav2

`semantic_costmap_node` uses `CameraInfo` and TF rather than dataset-specific
pixel coordinates. It publishes:

- `semantic_mask` as `sensor_msgs/Image`;
- `painted_points` as `sensor_msgs/PointCloud2`;
- `semantic_costmap` as `nav_msgs/OccupancyGrid`.

The C++ `semantic_costmap_layer` subscribes to an occupancy grid, transforms
cells into the Nav2 master frame, and performs a maximum-cost merge. A normal
Nav2 stack composes layers as:

```text
static map + obstacle/voxel layer + semantic layer + inflation layer
```

Thus this project contributes context-aware costs while Nav2 continues to own
map storage, inflation, and path-planner interfaces.

The supplied Nav2 parameter template places the standard `InflationLayer`
after the semantic layer. Inflation expands lethal costs around obstacles by a
configurable robot-safety radius rather than duplicating that algorithm inside
the custom plugin.

## 8. Pose and SLAM accumulation

SLAM Toolbox remains an upstream pose provider; this project does not implement
scan matching or loop closure. The accumulator looks up the observation pose
through the standard `map -> odom -> base_link` TF chain and places each local
observation into a global metric grid.

Static evidence uses maximum-cost persistence. Dynamic-obstacle evidence is
stored separately and expires after a configurable timeout, revealing any
underlying static cost. This avoids permanently painting a moving vehicle into
the map.

## 9. Safety and engineering boundaries

- Unknown space is preserved instead of treated as free.
- Semantic evidence never lowers raw obstacle cost.
- Stale sensor pairs and stale Nav2 semantic grids are rejected.
- Dynamic observations decay; static observations persist.
- Checkpoints, datasets, and generated outputs stay outside Git.
- Every math-heavy stage has focused tests and an inspectable offline demo.

The system is suitable as a research and portfolio pipeline. Deployment on a
robot still requires sensor-specific synchronization, TF validation, GPU or
optimized inference, field testing, and safety supervision.
