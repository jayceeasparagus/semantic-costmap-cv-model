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

Results are written to `outputs/playback/benchmark.json`. The benchmark reports
load, neural-network inference, calibration projection, point fusion, costmap,
and total latency separately. CPU timing documents correctness and a baseline;
GPU timing should be measured on the target deployment hardware before making a
real-time claim.
