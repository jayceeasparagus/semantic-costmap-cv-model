# End-to-end runbook

This project has one primary demonstration:

1. the offline A2D2 semantic-mapping pipeline, which runs the trained U-Net, calibrates LiDAR
   points, paints them with semantic predictions, and rasterizes a local
   metric semantic grid and persistent map.

## Offline pipeline

From the repository root, activate the virtual environment and run:

```bash
source .venv/bin/activate
export PYTHONPATH=src

python tools/run_inference.py --device cpu
python tools/validate_calibration_projection.py
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
python tools/run_playback.py --device cpu --stride 5 --max-frames 120 --gif-fps 10
```

The commands write diagnostic images and arrays under `outputs/`. The
checkpoint is intentionally ignored by Git and must be restored locally at
`outputs/checkpoints/epoch29_restore/best_semantic_unet.pt`.

For pose-aware accumulation, first build the pose CSV from the A2D2 bus
signals, then pass it to playback:

```bash
python tools/build_a2d2_poses.py
python tools/run_playback.py --device cpu --stride 5 --max-frames 120 \
  --gif-fps 10 \
  --poses-csv outputs/poses/20180807_bus_odometry.csv
```

## What is and is not demonstrated

- Camera semantics identify the class of image pixels; LiDAR supplies metric
  depth and 3D obstacle evidence.
- Calibration maps LiDAR returns into the camera image, and the painted labels
  are rasterized into a vehicle-frame grid.
- Ray tracing marks observed free space and obstacle footprints improve sparse
  returns.
- Pose-aware accumulation uses recorded bus-derived odometry; it is not a
  claim of localization accuracy.
- The measured local CPU playback rate and model metrics in
  `docs/benchmark_results.md` are the project’s reported performance numbers;
  no real-time claim is made for an arbitrary robot computer.
