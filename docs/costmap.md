# Offline semantic costmap

The offline grid uses the vehicle frame: `+x` is forward and `+y` is left. The
default map covers 50 m ahead and 20 m to either side at 0.20 m per cell.

Each painted point contributes its non-background class probabilities to a
cell. The cell cost is the probability-weighted average of the four navigation
costs. Background, low-confidence predictions, and out-of-range semantic
points do not directly contribute a class cost.

Two conservative density steps run after semantic aggregation:

1. Unknown cells surrounded by enough drivable neighbors are filled for a
   configurable number of iterations.
2. A vectorized grid ray from the sensor to every LiDAR return marks previously
   unknown cells before the endpoint as observed free space.

Ray tracing uses physical visibility, not model confidence. It never overwrites
an existing semantic cost and does not clear the return cell itself. On the
included sample, known coverage increases from 4.65% to 45.40%.

Raw LiDAR points inside the configured robot-height band are inserted as lethal
obstacles (`254`) after semantic aggregation. This ordering is intentional:
camera semantics, interpolation, and ray clearing may add context but can never
erase geometric collision evidence.

Generate the map after point painting:

```bash
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
```

Use `--disable-raytracing` or `--disable-ground-interpolation` for ablation
tests. Ray range, interpolation iterations, and neighbor threshold are also
available as command-line options and ROS parameters.

The command writes compressed NumPy arrays, a raw PGM plus YAML metadata, and a
color preview under `outputs/costmap/`. These generated files are ignored by
Git.
