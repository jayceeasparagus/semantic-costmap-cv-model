# Benchmark results

## Eight-frame CPU playback

Measured in WSL2 on an Intel Core Ultra 9 288V using PyTorch 2.13.0 and the
epoch-29 U-Net checkpoint. The model ran on CPU with full 1920x1208 output
probabilities and a 250x200 cost grid.

| Stage | Mean | 95th percentile |
|---|---:|---:|
| File loading | 104.0 ms | 113.9 ms |
| U-Net inference | 1155.5 ms | 1472.9 ms |
| Calibration projection | 4.4 ms | 5.8 ms |
| Semantic point painting | 3.3 ms | 4.3 ms |
| Costmap generation | 2.6 ms | 3.4 ms |
| End to end | 1269.7 ms | 1591.0 ms |

Mean throughput was **0.79 FPS**. The fusion and costmap stages together take
less than 7 ms on this CPU; neural-network inference is the clear bottleneck.
This is a reproducible CPU baseline, not a claim of deployment real-time speed.
GPU inference, model export, or a smaller architecture are the natural next
optimization targets if a higher frame rate is required.
