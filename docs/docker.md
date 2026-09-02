# Docker

The CPU image records the ROS 2 Jazzy, Nav2, SLAM Toolbox, Python, and PyTorch
runtime without embedding datasets or the trained checkpoint.

```bash
docker build -t semantic-costmap .
docker run --rm -it \
  --network host \
  -v "$PWD/data:/workspace/data:ro" \
  -v "$PWD/outputs:/workspace/outputs" \
  semantic-costmap
```

The checkpoint must exist on the host at
`outputs/checkpoints/epoch29_restore/best_semantic_unet.pt`. Start the configured
ROS nodes with `docker compose up --build`. Sensor topics and TF must be supplied
by the host or a rosbag. The image defaults to CPU inference so it works without
vendor-specific GPU container support.
