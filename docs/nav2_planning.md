# Nav2 planning demonstration

The repository includes a deterministic headless planner test. It starts a
free map from `nav2_demo_map.yaml`, publishes a switchable semantic barrier,
loads the custom `SemanticCostmapLayer`, applies Nav2's standard
`InflationLayer`, and runs Navfn through `ComputePathToPose`.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/semantic_ros/setup.bash
ros2 launch semantic_costmap_ros nav2_demo.launch.py
```

The client first requests a baseline path with the semantic barrier disabled,
then enables the barrier and requests a second path. It verifies that the
second path reaches the goal, does not enter a lethal semantic cell, and is
longer than the baseline. Generated evidence is written to
`outputs/nav2_demo/`:

- `path_without_semantics.json`;
- `path_with_semantics.json`;
- `planning_result.json`;
- `path_overlay.png` (blue baseline, green hazard-aware path, red hazard).

This is a planner integration test, not a replacement for Nav2's production
behavior tree or controller. A real robot would feed the semantic grid into
the same layered costmap and use Nav2's normal navigation actions.
