# Benchmark results

## Eight-frame CPU playback

Measured in WSL2 on an Intel Core Ultra 9 288V using PyTorch 2.13.0 and the
epoch-29 U-Net checkpoint. The model ran on CPU with full 1920x1208 output
probabilities and a 250x200 cost grid.

| Stage | Mean | 95th percentile |
|---|---:|---:|
| File loading | 98.4 ms | 125.5 ms |
| U-Net inference | 1535.4 ms | 1841.1 ms |
| Calibration projection | 7.0 ms | 9.9 ms |
| Semantic point painting | 4.0 ms | 8.0 ms |
| Dense costmap generation | 19.4 ms | 27.0 ms |
| End to end | 1664.3 ms | 1977.6 ms |

Mean throughput was **0.60 FPS**. Projection, painting, and dense costmap
generation—including interpolation and free-space ray tracing—took 30.4 ms
combined. Neural-network inference remains the clear bottleneck. CPU timing is
sensitive to system load, so this is a reproducible baseline rather than a
real-time claim. GPU inference, model export, or a smaller architecture are the
natural next optimization targets if a higher frame rate is required.
