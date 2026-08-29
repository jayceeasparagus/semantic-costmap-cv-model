# System architecture

Status: accepted initial architecture

## 1. Project definition

The project is a semantic LiDAR costmap pipeline. An RGB model predicts dense
semantic probabilities, calibrated LiDAR supplies metric position, and a custom
Nav2 layer turns the fused observations into navigation costs.

The segmentation model does not directly output a costmap. The complete
perception pipeline produces the costmap.

## 2. Scope

The initial system includes:

- a six-class U-Net-style semantic segmentation model trained on A2D2;
- synchronized RGB and LiDAR ingestion;
- camera-LiDAR projection and semantic point painting;
- confidence-aware bird's-eye-view rasterization;
- temporal evidence accumulation using externally supplied poses;
- ROS 2 replay and perception nodes;
- a custom Nav2 costmap layer;
- RViz debugging, accuracy evaluation, and runtime benchmarks.

The initial system does not include monocular depth estimation, a learned LiDAR
network, a custom SLAM implementation, 3D detection/tracking, vehicle control,
or multi-camera fusion.

## 3. End-to-end flow

```text
OFFLINE TRAINING
A2D2 RGB + source labels
        -> compact six-class masks
        -> U-Net training and validation
        -> best model checkpoint

RUNTIME
RGB image
        -> model logits
        -> per-pixel class probabilities --------+
                                                   |
LiDAR + camera calibration ------------------------+
        -> project points into image
        -> sample semantic probabilities
        -> painted point cloud
        -> bird's-eye semantic evidence grid
        -> semantic Nav2 costmap layer

Raw LiDAR
        -> standard Nav2 obstacle or voxel layer

SLAM/localization + odometry
        -> timestamped transforms
        -> persistent placement of observations

standard layer + semantic layer + inflation layer
        -> combined Nav2 costmap
        -> RViz and planner
```

## 4. Semantic contract

`configs/semantic_classes.yaml` is the single source of truth for model output
IDs, source-label grouping, visualization colors, navigation costs, and temporal
policies.

The model has six output channels:

| ID | Class | Navigation meaning | Default cost |
|---:|---|---|---:|
| 0 | drivable | preferred traversable surface | 0 |
| 1 | caution | traversable but less preferred | 80 |
| 2 | non_drivable | strongly avoided surface | 220 |
| 3 | static_obstacle | fixed collision hazard | 254 |
| 4 | dynamic_obstacle | moving or potentially moving hazard | 254 |
| 5 | background | visible context that is never mapped | none |

Training-mask ID 255 means ignore and is excluded from loss and metrics. Nav2
cost 255 means unknown space. These values share a number but not a meaning and
must remain separate in code.

## 5. Segmentation model

The initial model is a U-Net-style encoder-decoder implemented in PyTorch. The
encoder extracts increasingly abstract features, the decoder restores spatial
resolution, and skip connections preserve boundaries and small objects.

The final layer emits six logits per pixel. Softmax probabilities are retained
for fusion rather than immediately reducing every pixel to an argmax class.

Training uses sequence- or scene-level splits to avoid temporal leakage. Primary
metrics are per-class IoU, all-class mIoU, navigation-only mIoU, model size,
latency, and throughput.

## 6. Sensor synchronization

A fusion update requires an RGB frame and LiDAR cloud with compatible capture
times. Offline code pairs records by dataset timestamps. ROS 2 code uses
timestamped messages and an approximate-time synchronizer with matching QoS.
Headerless or arrival-time synchronization is not accepted.

Every transform lookup uses the sensor message timestamp, not the most recent
available transform.

## 7. Camera-LiDAR projection

For a homogeneous LiDAR point `p_L`, extrinsic calibration gives its position in
the camera frame:

```text
p_C = T_C_L p_L
```

Points behind the camera or outside the valid sensor range are discarded. For a
rectified pinhole image, a camera-frame point `(X, Y, Z)` projects to:

```text
u = fx X / Z + cx
v = fy Y / Z + cy
```

Distorted images must be rectified first or projected with the full distortion
model. Projection is verified visually and, where A2D2 supplies image
coordinates, numerically against the dataset registration.

## 8. Semantic point painting

For each valid projected point, the fusion module samples the model probability
vector at pixel `(u, v)` and appends it to the point:

```text
[x, y, z, intensity, P0, P1, P2, P3, P4, P5]
```

This sequential camera-LiDAR fusion method is semantic point painting. A small
pixel-space depth buffer rejects farther points that would otherwise inherit the
semantic class of an occluding foreground object.

Low-confidence or background predictions do not mark free space. They produce no
semantic update.

## 9. Bird's-eye semantic grid

Painted points are transformed to the active costmap frame and rasterized by
metric position:

```text
cell_x = floor((point_x - origin_x) / resolution)
cell_y = floor((point_y - origin_y) / resolution)
```

For every observed cell, class evidence is accumulated from point probabilities:

```text
evidence[cell, class] += observation_weight * P(class | point)
```

Normalized evidence produces a cell probability distribution. The initial
graded cost is the probability-weighted sum of configured class costs. Static or
dynamic obstacle evidence above a configurable threshold produces lethal cost.

Multiple observations may increase a cell's cost, but semantic inference may
never clear a geometric obstacle.

## 10. Nav2 layer composition

The intended Nav2 order is:

```text
StaticLayer
ObstacleLayer or VoxelLayer       # raw LiDAR geometry and ray clearing
SemanticCostmapLayer              # painted LiDAR navigation meaning
InflationLayer                    # robot footprint margin
```

The custom C++ `SemanticCostmapLayer` inherits from
`nav2_costmap_2d::Layer`, subscribes to painted points, tracks changed bounds,
and merges semantic costs conservatively into the master grid.

A debug occupancy grid may be published for RViz, but it is not the primary
Nav2 integration interface.

## 11. Temporal fusion

Each observation is transformed using the robot pose at its timestamp. Initial
temporal fusion uses decayed evidence:

```text
evidence_t = decay * evidence_(t-1) + current_observation
```

Surface and static-obstacle evidence persists longer than dynamic-obstacle
evidence. Dynamic observations expire quickly to avoid ghost obstacles. Standard
LiDAR ray clearing remains responsible for geometric free-space updates.

## 12. Coordinate frames

The required frame tree is:

```text
map -> odom -> base_link -> camera_link -> camera_optical_frame
                         -> lidar_link
```

Responsibilities:

- SLAM or localization publishes `map -> odom`;
- odometry publishes `odom -> base_link`;
- calibrated static transforms connect `base_link` to each sensor;
- the semantic mapper consumes these transforms but does not estimate them.

ROS REP-103/105 conventions and SI units are used throughout.

## 13. ROS 2 components and topics

### A2D2 replay node

Publishes dataset records as robot-like sensor streams:

- `/camera/front/image_rect` (`sensor_msgs/Image`)
- `/camera/front/camera_info` (`sensor_msgs/CameraInfo`)
- `/lidar/front/points` (`sensor_msgs/PointCloud2`)
- `/odom` and `/tf` when pose data is available

### Semantic fusion node

Runs model inference and point painting. It publishes:

- `/semantic/mask` for class-ID visualization;
- `/semantic/confidence` for uncertainty inspection;
- `/semantic/painted_points` with class-score fields;
- `/semantic/projection_overlay` for calibration debugging.

### Semantic costmap layer

Consumes painted points, transforms them into the costmap frame, accumulates
evidence, applies temporal policies, and updates the Nav2 master costmap.

## 14. Package boundaries

```text
src/semantic_costmap/
  data/          A2D2 records, labels, manifests, and splits
  models/        U-Net definition
  training/      losses, loops, checkpoints, and metrics
  inference/     preprocessing and model runtime
  geometry/      calibration, transforms, and projection
  fusion/        point painting and temporal evidence
  costmap/       grid math and cost policy reference implementation
  evaluation/    segmentation, projection, map, and latency metrics

ros2_ws/src/
  semantic_costmap_ros/    replay and fusion nodes
  semantic_costmap_layer/  C++ Nav2 plugin
```

Notebooks orchestrate library code; they do not own reusable implementation.
Large data, checkpoints, generated maps, and videos remain outside Git.

## 15. Safety and correctness invariants

1. Raw LiDAR geometry overrides semantic free-space predictions.
2. Background and low-confidence predictions never clear cells.
3. Missing transforms or stale synchronized data cause an update to be dropped.
4. Unknown space is not silently converted to drivable space.
5. Every model output ID is defined in the semantic configuration.
6. Every A2D2 source label is mapped exactly once or explicitly ignored.
7. Dynamic evidence expires; static evidence cannot persist without bounds.
8. Model, projection, fusion, and ROS wrappers are tested independently.

## 16. Validation ladder

Development proceeds in independently verifiable stages:

1. validate the taxonomy against all A2D2 labels;
2. train and evaluate segmentation using scene-level splits;
3. validate projection using ground-truth masks before model predictions;
4. paint one LiDAR frame and render it in image and bird's-eye views;
5. generate a single-frame semantic cost grid;
6. accumulate a sequence using supplied poses;
7. replay the sequence through ROS 2;
8. integrate the custom layer with Nav2 and RViz;
9. benchmark accuracy, latency, memory, and temporal stability;
10. add Docker packaging and continuous integration.

Ground-truth-mask projection is a required geometry test, not a disposable
prototype. It separates calibration defects from model errors.

## 17. References

- PointPainting: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.pdf>
- Nav2 costmap plugin guide: <https://docs.nav2.org/plugin_tutorials/docs/writing_new_costmap2d_plugin.html>
- Nav2 transform setup: <https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/transformation/setup_transforms/>
- ROS 2 approximate synchronization: <https://docs.ros.org/en/ros2_packages/jazzy/api/message_filters/doc/Tutorials/Approximate-Synchronizer-Cpp.html>
- A2D2 dataset paper: <https://www.a2d2.audi/content/dam/a2d2/dataset/a2d2-audi-autonomous-driving-dataset.pdf>
