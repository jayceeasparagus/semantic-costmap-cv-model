# Navigation semantic taxonomy

The model predicts six compact classes rather than all 55 A2D2 labels. The
groups are selected by how a navigation system should react, not only by visual
appearance.

| ID | Model class | Nav2 cost | Meaning |
|---:|---|---:|---|
| 0 | drivable | 0 | ordinary traversable road surface |
| 1 | caution | 80 | traversable but deserving slower or less-preferred motion |
| 2 | non_drivable | 220 | surface the planner should strongly avoid |
| 3 | static_obstacle | 254 | fixed lethal obstacle |
| 4 | dynamic_obstacle | 254 | moving or potentially moving lethal obstacle |
| 5 | background | none | visible context such as sky that is never projected as cost |
| 255 | ignore | none | invalid, obscured, or ego-vehicle pixels excluded from loss |

Class 5 and ID 255 have different roles. Background is learned so the network
can recognize sky reliably; ignore pixels contribute neither loss nor metrics.

Static and dynamic obstacles initially receive the same lethal cost. They remain
separate because later temporal fusion can retain static observations longer and
clear dynamic observations sooner.

The YAML file is the source of truth shared by label conversion, training,
evaluation, and future costmap generation. Run this after changing it:

```bash
python tools/validate_a2d2_mapping.py
```
