# System architecture

## 1. Purpose

The pipeline converts synchronized RGB and LiDAR observations into a persistent
semantic map. It separates learned perception, sensor geometry, and temporal
map fusion so each stage can be inspected and tested independently.

## 2. End-to-end data flow

```text
RGB image ---------------------> U-Net -----> H x W x 5 probabilities
                                                      |
LiDAR points -> TF/extrinsics -> camera frame --------+-> point painting
CameraInfo/calibration --------> pixel projection ----+       |
                                                              v
                                               XYZ + class probabilities
                                                              |
                                               frame-level rasterization
                                                              |
                                                   semantic grid + confidence
                                                                   |
                                                         pose-aware accumulator
                                                                   |
                                                         persistent global map
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

## 5. Frame-level semantic grid

Camera-painted points are transformed into the vehicle frame, filtered by
height and confidence, and assigned to grid cells. This grid is an
intermediate representation for mapping, not a separate navigation product:

```text
cell_x = floor((point_x - origin_x) / resolution)
cell_y = floor((point_y - origin_y) / resolution)
```

Each cell averages its accumulated class probabilities, weighted by the
confidence of each painted point, and converts the result to a
probability-weighted navigation cost. The local grid stores a fused confidence
value alongside the class and cost. Small unknown gaps are filled only when
they have enough neighboring drivable cells. This is local interpolation, not
an assumption that every unknown cell is road.

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
Offline file playback is the primary runtime path: it makes the data flow
deterministic and keeps the project easy to reproduce without middleware.

## 7. Pose-aware accumulation

`tools/build_a2d2_poses.py` reads the actual A2D2 list-of-frame-records bus JSON.
It aligns `vehicle_speed` and `angular_velocity_omega_z` to each camera
timestamp, converts km/h and degrees/s to SI units, and integrates a planar
unicycle model. The output is explicitly **bus-derived odometry**, not
ground-truth localization. A camera metadata directory can be supplied so `cam_tstamp` is
used instead of the bus record timestamp. The initial yaw is an explicit
assumption because GPS samples alone do not provide a reliable local heading
in this small replay.

The accumulator consumes the recorded pose from the CSV and places each local
observation into a global metric grid. It does not implement scan matching or
loop closure. Offline playback saves both
`accumulated_costmap_preview.png` and `odometry_trajectory.png`.

Static evidence uses confidence-weighted semantic voting plus conservative
maximum-cost persistence. Dynamic-obstacle evidence is stored separately,
retains a confidence value, and expires after a configurable timeout, revealing
any underlying static cost. This avoids permanently painting a moving vehicle
into the map. The offline playback also exports a confidence heatmap and
reports mean map confidence and the number of currently active dynamic cells.

## 8. Safety and engineering boundaries

- Unknown space is preserved instead of treated as free.
- Semantic evidence never lowers raw obstacle cost.
- Stale sensor pairs are rejected.
- Dynamic observations decay; static observations persist.
- Checkpoints, datasets, and generated outputs stay outside Git.
- Every math-heavy stage has focused tests and an inspectable offline demo.

The system is suitable as a research and portfolio pipeline. Deployment on a
robot still requires sensor-specific synchronization, TF validation, GPU or
optimized inference, field testing, and safety supervision.
