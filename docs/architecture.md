# Project architecture

This document describes the intended system without committing the repository
to implementation details too early.

## 1. Inputs and output

Inputs:

- synchronized front-camera RGB frames;
- synchronized LiDAR point clouds;
- camera intrinsics and camera-to-LiDAR extrinsics;
- timestamped robot poses when temporal mapping is introduced.

Output:

- a two-dimensional grid whose cells contain Nav2-compatible navigation costs.

## 2. Offline model training

A U-Net-style model was trained from scratch on A2D2 RGB images and reduced
semantic masks. Its output is a five-channel logit tensor. Softmax converts those
logits into a probability for each class at every pixel.

The proposed classes are:

| ID | Class | Navigation meaning | Initial cost |
|---:|---|---|---:|
| 0 | drivable | preferred surface | 0 |
| 1 | non_drivable | strongly avoid | 220 |
| 2 | static_obstacle | fixed collision hazard | 254 |
| 3 | dynamic_obstacle | moving collision hazard | 254 |
| 4 | background | never add to the grid | none |

The A2D2 label audit removed the earlier `caution` class because it was too rare
to learn reliably. Keeping five well-supported classes gives the costmap clearer
and more defensible semantics.

## 3. Camera-LiDAR connection: semantic point painting

The natural fusion algorithm for this project is **semantic point painting**.
It is sequential fusion: infer camera semantics first, then attach those
semantics to geometrically aligned LiDAR points.

For each LiDAR point:

1. Transform it from the LiDAR frame to the camera frame using the calibrated
   extrinsic transform.
2. Reject points behind the camera.
3. Project the 3D camera-frame point into image coordinates using the camera
   intrinsics.
4. Reject points outside the image.
5. Sample the model's five probabilities at that image pixel.
6. Store the 3D point together with its semantic probabilities.

In compact form:

```text
p_camera = T_camera_lidar * p_lidar
u = fx * X / Z + cx
v = fy * Y / Z + cy
painted_point = [x, y, z, P(class 0), ..., P(class 4)]
```

This is more defensible than estimating distance from the RGB image because the
dataset already provides measured LiDAR geometry and calibration.

## 4. Building the cost grid

Painted points are transformed into the grid frame and assigned to cells:

```text
cell_x = floor((point_x - origin_x) / resolution)
cell_y = floor((point_y - origin_y) / resolution)
```

Each cell aggregates the semantic evidence from its points. We will begin with
a simple, inspectable rule such as the highest-confidence valid class or a
probability-weighted cost. Later experiments can compare aggregation rules.

Important safety rule: semantic predictions may increase navigation cost, but
they may not erase an obstacle detected by raw LiDAR geometry.

## 5. ROS 2 and Nav2

Nav2 already provides map storage, obstacle layers, inflation, coordinate-frame
handling, and planner interfaces. This project should contribute a semantic
layer, not recreate the navigation stack.

The intended layer composition is:

```text
existing static map
    + raw LiDAR obstacle/voxel layer
    + this project's semantic cost layer
    + Nav2 inflation layer
    = planner-ready costmap
```

The first cost grid can be generated offline in Python. Once its math is tested,
a ROS 2 node will publish debug products and a Nav2 costmap plugin will merge
semantic costs into Nav2's master grid.

## 6. Where SLAM belongs

SLAM provides the robot pose and the `map -> odom` relationship. It does not
produce camera semantics. This project will consume poses from an existing SLAM
system, such as SLAM Toolbox, rather than implement SLAM.

Development should begin with a robot-relative rolling costmap, which needs no
persistent global map. Persistent accumulation comes later, after a data source
with valid odometry/transforms is available. A2D2 is useful for training and
sensor-fusion experiments, but it does not need to serve every ROS integration
stage.

## 7. Planned implementation phases

### Phase A: segmentation

Create paired RGB/mask data, train U-Net, inspect predictions, and report
per-class IoU and mIoU.

### Phase B: geometric fusion

Load one RGB/LiDAR/calibration sample, verify projection visually, then paint
LiDAR points with ground-truth semantics before using model predictions.

### Phase C: costmap

Rasterize painted points into a robot-relative grid and validate orientation,
resolution, costs, and obstacle precedence.

### Phase D: temporal and ROS integration

Use timestamped transforms to place observations over time, publish ROS 2 debug
topics, and integrate a semantic layer with Nav2.

### Phase E: engineering finish

Add automated tests, accuracy/runtime benchmarks, Docker packaging, RViz demos,
and final documentation only after the pipeline works end to end.

## 8. Success criteria

The finished project should demonstrate:

- measurable segmentation quality, including per-class IoU;
- visibly correct camera-LiDAR projection;
- correct metric placement and semantic costs;
- conservative interaction with raw LiDAR obstacles;
- repeatable ROS 2/Nav2 playback;
- measured end-to-end latency and update rate.
