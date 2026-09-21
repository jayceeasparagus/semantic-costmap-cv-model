# Multi-frame semantic mapping playback and benchmarks

`tools/run_playback.py` keeps the model and calibration loaded while processing
paired A2D2 frames. Each rendered frame shows the RGB image, semantic overlay,
painted LiDAR returns, and a frame-level semantic grid. With poses, the same
frames are fused into a persistent map. The tool saves an animated GIF,
individual PNG frames, and machine-readable latency results.

Expected local data layout:

```text
data/raw/sequential_playback/camera/*.png
data/raw/sequential_playback/lidar/*.npz
```

Run a CPU benchmark with:

```bash
python tools/run_playback.py --device cpu --stride 3 --max-frames 200 --gif-fps 10
```

To accumulate the frames in a persistent map, provide map-to-base poses:

```csv
frame_id,timestamp,x,y,yaw
000000091,0.0,0.0,0.0,0.0
000000127,0.1,1.0,0.0,0.0
```

```bash
python tools/run_playback.py \
  --device cpu \
  --stride 3 \
  --max-frames 200 \
  --gif-fps 10 \
  --poses-csv poses.csv
```

The frame ID must match the nine-digit ID in each A2D2 filename. `x`, `y`, and
`yaw` describe the map-to-base pose in meters and radians. A real run should
export these poses from recorded odometry or localization; synthetic values
are appropriate only for checking coordinate placement.

Results are written to `outputs/playback/benchmark.json`. The benchmark reports
load, neural-network inference, calibration projection, point fusion, costmap,
and total latency separately. CPU timing documents correctness and a baseline;
GPU timing should be measured on the target deployment hardware before making a
real-time claim.

With poses, the tool additionally saves `accumulated_costmap.npz` and
`accumulated_costmap_preview.png`, `accumulated_confidence.png`, and
`odometry_trajectory.png` in the selected output directory.
The PNG previews are cropped around known cells for easier viewing; the NPZ
file retains the full configured map bounds.

The default command measures mapping only. The experimental frame-level route
planner is intentionally opt-in:

```bash
python tools/run_playback.py --device cpu --max-frames 8 --plan-route
```

Route-planning timings are included in the benchmark only when this flag is
used, so the main benchmark describes the semantic-mapping pipeline directly.
