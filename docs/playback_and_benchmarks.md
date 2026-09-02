# Multi-frame playback and benchmarks

`tools/run_playback.py` keeps the model and calibration loaded while processing
paired A2D2 frames. Each rendered frame shows the RGB image, semantic overlay,
painted LiDAR returns, and vehicle-relative costmap. The tool saves an animated
GIF, individual PNG frames, and machine-readable latency results.

Expected local data layout:

```text
data/raw/a2d2_playback/camera/*.png
data/raw/a2d2_playback/lidar/*.npz
```

Run a CPU benchmark with:

```bash
python tools/run_playback.py --device cpu --max-frames 8
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
  --max-frames 8 \
  --poses-csv poses.csv
```

The frame ID must match the nine-digit ID in each A2D2 filename. `x`, `y`, and
`yaw` describe the map-to-base pose in meters and radians. A real run should
export these poses from odometry, localization, or SLAM; synthetic values are
appropriate only for checking coordinate placement.

Results are written to `outputs/playback/benchmark.json`. The benchmark reports
load, neural-network inference, calibration projection, point fusion, costmap,
and total latency separately. CPU timing documents correctness and a baseline;
GPU timing should be measured on the target deployment hardware before making a
real-time claim.

With poses, the tool additionally saves `accumulated_costmap.npz` and
`accumulated_costmap_preview.png` in the selected output directory.
