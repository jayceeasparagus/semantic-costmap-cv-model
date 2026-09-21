# Semantic Mapping from Camera and LiDAR

An undergraduate-level autonomous-vehicle perception project that builds a
persistent semantic map while a vehicle moves through an environment.

```text
RGB frame -> U-Net -> per-pixel semantic probabilities
                                  |
LiDAR + camera calibration -------+-> semantically painted 3D points
                                         |
                                         v
                                frame-level semantic grid
                                         |
                              pose-aligned map fusion
                                         |
                              persistent semantic map
```

The camera predicts **what** is present, LiDAR measures **where** it is, and
vehicle poses place observations into a shared map frame. The project is an
offline A2D2 sequence replay focused on semantic map construction.

## Project scope

The main question is: can synchronized camera and LiDAR observations be fused
into a useful semantic map of the road environment? The project focuses on
drivable ground, non-drivable regions, static obstacles, dynamic obstacles,
and unknown space. Route planning is kept as an optional experiment and is not
part of the main mapping benchmark.

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

## Semantic classes

| ID | Class | Mapping meaning |
|---:|---|---|
| 0 | `drivable` | preferred ground |
| 1 | `non_drivable` | strongly avoid |
| 2 | `static_obstacle` | fixed collision hazard |
| 3 | `dynamic_obstacle` | temporary collision hazard |
| 4 | `background` | sky and non-spatial context |

Raw LiDAR obstacle evidence can raise a cell's cost but semantic predictions
cannot lower it. Cells outside observed rays remain unknown.

LiDAR ray tracing marks observed space before each return as free, and a
conservative neighbor rule fills small gaps surrounded by drivable evidence.
On the included sample these steps increased single-frame known coverage from
4.65% to 45.40% without lowering obstacle costs.

## Local setup

The tools require Python 3.10 or newer:

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

# 4. Rasterize the painted cloud into a frame-level metric grid.
python tools/generate_costmap.py

# 5. Replay a longer sequence while skipping intermediate frames.
python tools/run_playback.py --device cpu --stride 5 --max-frames 60 --gif-fps 10

# 6. Accumulate frame-level grids into a persistent map.
python tools/demo_pose_accumulation.py
```

For pose-aligned sequence accumulation, provide a CSV containing
`frame_id,timestamp,x,y,yaw` and add `--poses-csv path/to/poses.csv` to the
playback command. These are map-to-base poses from recorded odometry or another
localization source.

Each tool writes inspectable images and arrays under `outputs/`. The playback
GIF shows the RGB image, semantic overlay, painted LiDAR, and frame-level
semantic grid. When poses are supplied, it also writes the persistent semantic
map, confidence map, and trajectory preview. Detailed data flow and equations
are in [the architecture guide](docs/architecture.md).

## Project boundaries

The repository intentionally focuses on offline semantic mapping. The mapping
demo uses recorded odometry poses to place observations in a shared frame.

## Tests

```bash
source .venv/bin/activate
tools/run_checks.sh
```

The check runs the Python tests and compile checks. The A2D2 integration test
skips when local data or the checkpoint is unavailable.

## Repository layout

```text
configs/                 A2D2 calibration and label definitions
docs/                    Design, demo, and test documentation
src/semantic_costmap/    Shared inference, geometry, fusion, and mapping code
tools/                   Offline demos and validation commands
tests/                   Unit and end-to-end integration tests
data/                    Local datasets (ignored)
outputs/                 Generated artifacts and checkpoints (ignored)
```

## Scope and limitations

This is a tested offline mapping pipeline, not a vehicle-certified system. The
main map demo uses bus-derived odometry, so it should be evaluated as a
relative mapping demonstration rather than a localization benchmark. GPU
inference or model optimization is needed for practical real-time rates.
