import math

import numpy as np

from semantic_costmap_ros.pointcloud_to_laserscan_node import points_to_ranges


def test_nearest_valid_point_wins_each_laser_bin():
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]
    )
    ranges = points_to_ranges(
        points,
        min_height=-1.0,
        max_height=1.0,
        angle_min=-math.pi,
        angle_max=math.pi,
        angle_increment=math.pi / 2.0,
        range_min=0.3,
        range_max=10.0,
    )

    assert ranges.shape == (4,)
    assert ranges[2] == 1.0
    assert ranges[3] == 2.0
    assert np.isinf(ranges[:2]).all()
