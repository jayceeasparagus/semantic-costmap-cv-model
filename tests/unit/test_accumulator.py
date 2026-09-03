import numpy as np

from semantic_costmap.mapping import GlobalMapConfig, PoseAwareAccumulator


def make_accumulator(decay=2.0):
    return PoseAwareAccumulator(
        GlobalMapConfig(
            resolution=1.0,
            x_min=-5.0,
            x_max=5.0,
            y_min=-5.0,
            y_max=5.0,
            dynamic_decay_seconds=decay,
        )
    )


def test_static_observations_persist_and_take_maximum_cost():
    accumulator = make_accumulator()
    point = np.array([[1.2, 2.4]])
    accumulator.update_map_points(point, [0], [0], timestamp=0.0)
    accumulator.update_map_points(point, [1], [220], timestamp=1.0)

    grid = accumulator.grid(timestamp=100.0)
    assert grid[7, 6] == 220


def test_dynamic_observations_expire_without_erasing_static_map():
    accumulator = make_accumulator(decay=2.0)
    point = np.array([[0.2, 0.2]])
    accumulator.update_map_points(point, [0], [0], timestamp=0.0)
    accumulator.update_map_points(point, [3], [254], timestamp=1.0)

    assert accumulator.grid(timestamp=2.0)[5, 5] == 254
    assert accumulator.grid(timestamp=4.0)[5, 5] == 0


def test_background_points_are_not_inserted():
    accumulator = make_accumulator()
    accumulator.update_map_points(
        np.array([[0.2, 0.2]]),
        [4],
        [-1],
        timestamp=0.0,
    )
    assert accumulator.grid(timestamp=0.0)[5, 5] == 255


def test_semantic_grid_preserves_class_ids_and_unknown_cells():
    accumulator = make_accumulator()
    accumulator.update_map_points(
        np.array([[0.2, 0.2], [1.2, 0.2]]),
        [0, 2],
        [0, 254],
        timestamp=0.0,
    )

    semantic = accumulator.semantic_grid(timestamp=0.0)
    known = accumulator.semantic_known_mask(timestamp=0.0)

    assert semantic[5, 5] == 0
    assert semantic[5, 6] == 2
    assert semantic[0, 0] == 4
    assert known[5, 5]
    assert not known[0, 0]


def test_out_of_bounds_points_are_ignored():
    accumulator = make_accumulator()
    accumulator.update_map_points(
        np.array([[-100.0, 0.0], [0.2, 0.2], [100.0, 0.0]]),
        [2, 0, 3],
        [254, 0, 254],
        timestamp=0.0,
    )

    semantic = accumulator.semantic_grid(timestamp=0.0)

    assert semantic[5, 5] == 0
    assert np.count_nonzero(accumulator.semantic_known_mask(0.0)) == 1
