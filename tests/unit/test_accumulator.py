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


def test_confidence_weighting_prefers_reliable_static_observation():
    accumulator = make_accumulator()
    point = np.array([[0.2, 0.2]])
    accumulator.update_map_points(
        np.repeat(point, 2, axis=0),
        [0, 2],
        [0, 254],
        timestamp=0.0,
        confidence=[0.9, 0.1],
    )

    semantic = accumulator.semantic_grid(timestamp=0.0)
    confidence = accumulator.confidence_grid(timestamp=0.0)

    assert semantic[5, 5] == 0
    assert confidence[5, 5] > 0.8
    # Costs remain conservative even when the semantic vote favors drivable.
    assert accumulator.grid(timestamp=0.0)[5, 5] == 254


def test_map_gap_fill_only_fills_drivable_holes():
    accumulator = PoseAwareAccumulator(
        GlobalMapConfig(
            resolution=1.0,
            x_min=-5.0,
            x_max=5.0,
            y_min=-5.0,
            y_max=5.0,
            map_gap_fill_iterations=1,
            map_gap_fill_min_neighbors=5,
        )
    )
    neighbors = np.array(
        [
            [0.2, 1.2],
            [1.2, 1.2],
            [2.2, 1.2],
            [0.2, 0.2],
            [2.2, 0.2],
            [0.2, -0.8],
            [1.2, -0.8],
            [2.2, -0.8],
        ]
    )
    accumulator.update_map_points(
        neighbors,
        np.zeros(len(neighbors), dtype=np.uint8),
        np.zeros(len(neighbors), dtype=np.int16),
        timestamp=0.0,
        confidence=np.ones(len(neighbors), dtype=np.float32),
    )

    semantic = accumulator.semantic_grid(timestamp=0.0)
    known = accumulator.semantic_known_mask(timestamp=0.0)

    assert semantic[5, 6] == 0
    assert known[5, 6]
    assert accumulator.confidence_grid(timestamp=0.0)[5, 6] == 0.5
