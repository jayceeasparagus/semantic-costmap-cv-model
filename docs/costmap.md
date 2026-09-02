# Offline semantic costmap

The offline grid uses the vehicle frame: `+x` is forward and `+y` is left. The
default map covers 50 m ahead and 20 m to either side at 0.20 m per cell.

Each painted point contributes its non-background class probabilities to a
cell. The cell cost is the probability-weighted average of the four navigation
costs. Background, low-confidence predictions, out-of-range points, and cells
without evidence remain unknown (`255`).

Raw LiDAR points inside the configured robot-height band are inserted as lethal
obstacles (`254`) after semantic aggregation. This ordering is intentional:
camera semantics may add context but can never erase geometric collision
evidence.

Generate the map after point painting:

```bash
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
```

The command writes compressed NumPy arrays, a raw PGM plus YAML metadata, and a
color preview under `outputs/costmap/`. These generated files are ignored by
Git.
