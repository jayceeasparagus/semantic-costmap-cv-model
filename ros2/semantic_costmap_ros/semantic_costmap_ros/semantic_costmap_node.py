"""ROS 2 node that fuses the latest camera semantics with a LiDAR cloud."""

from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from semantic_costmap.config import DEFAULT_CHECKPOINT_PATH
from semantic_costmap.costmap import CostmapConfig, build_semantic_costmap
from semantic_costmap.fusion import paint_points
from semantic_costmap.geometry import (
    project_camera_points,
    transform_points,
)
from semantic_costmap.inference import SegmentationResult, SemanticSegmenter
from semantic_costmap_ros.conversions import (
    image_message_to_pil,
    pointcloud_message_to_xyz,
    transform_message_to_matrix,
)


class SemanticCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_costmap_node")
        self.declare_parameter("checkpoint_path", str(DEFAULT_CHECKPOINT_PATH))
        self.declare_parameter("device", "auto")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("maximum_image_age", 0.15)
        self.declare_parameter("resolution", 0.20)
        self.declare_parameter("forward_range", 50.0)
        self.declare_parameter("side_range", 20.0)
        self.declare_parameter("minimum_confidence", 0.50)
        self.declare_parameter("raytrace_free_space", True)
        self.declare_parameter("raytrace_max_range", 50.0)
        self.declare_parameter("ground_interpolation_iterations", 2)
        self.declare_parameter("ground_interpolation_min_neighbors", 3)

        checkpoint_path = Path(
            self.get_parameter("checkpoint_path").get_parameter_value().string_value
        )
        device = self.get_parameter("device").value
        self.base_frame = self.get_parameter("base_frame").value
        self.maximum_image_age = float(self.get_parameter("maximum_image_age").value)
        self.segmenter = SemanticSegmenter(checkpoint_path, device)
        side_range = float(self.get_parameter("side_range").value)
        self.costmap_config = CostmapConfig(
            resolution=float(self.get_parameter("resolution").value),
            x_max=float(self.get_parameter("forward_range").value),
            y_min=-side_range,
            y_max=side_range,
            minimum_confidence=float(
                self.get_parameter("minimum_confidence").value
            ),
            raytrace_free_space=bool(
                self.get_parameter("raytrace_free_space").value
            ),
            raytrace_max_range=float(
                self.get_parameter("raytrace_max_range").value
            ),
            ground_interpolation_iterations=int(
                self.get_parameter("ground_interpolation_iterations").value
            ),
            ground_interpolation_min_neighbors=int(
                self.get_parameter("ground_interpolation_min_neighbors").value
            ),
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_info: CameraInfo | None = None
        # Inference is slower than replay on a CPU. Keep several completed
        # frames so a queued cloud is matched by timestamp instead of being
        # fused with whichever image callback happened to run most recently.
        self.segmentations: dict[int, tuple[SegmentationResult, Header]] = {}
        self.pending_clouds: dict[int, PointCloud2] = {}
        self.processed_clouds = 0

        self.image_group = MutuallyExclusiveCallbackGroup()
        self.cloud_group = MutuallyExclusiveCallbackGroup()
        self.info_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            CameraInfo,
            "camera_info",
            self._camera_info_callback,
            10,
            callback_group=self.info_group,
        )
        self.create_subscription(
            Image,
            "image",
            self._image_callback,
            2,
            callback_group=self.image_group,
        )
        self.create_subscription(
            PointCloud2,
            "points",
            self._pointcloud_callback,
            2,
            callback_group=self.cloud_group,
        )
        self.mask_publisher = self.create_publisher(Image, "semantic_mask", 2)
        self.painted_publisher = self.create_publisher(
            PointCloud2, "painted_points", 2
        )
        self.costmap_publisher = self.create_publisher(
            OccupancyGrid, "semantic_costmap", 2
        )
        self.get_logger().info(
            f"Loaded checkpoint epoch {self.segmenter.epoch} on {self.segmenter.device}"
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _image_callback(self, message: Image) -> None:
        try:
            image = image_message_to_pil(message)
            segmentation = self.segmenter.predict(image)
            stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
            self.segmentations[stamp_ns] = (segmentation, message.header)
            while len(self.segmentations) > 8:
                del self.segmentations[next(iter(self.segmentations))]
            self._publish_mask(segmentation, message.header)
            cloud = self.pending_clouds.pop(stamp_ns, None)
            if cloud is not None:
                self._process_cloud(cloud, segmentation)
        except (ValueError, RuntimeError) as error:
            self.get_logger().error(f"Image processing failed: {error}")

    def _lookup_matrix(self, target: str, source: str, stamp) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            target,
            source,
            Time.from_msg(stamp),
            timeout=Duration(seconds=0.1),
        )
        return transform_message_to_matrix(transform.transform)

    def _pointcloud_callback(self, message: PointCloud2) -> None:
        if self.camera_info is None:
            self.get_logger().warning("Waiting for CameraInfo", throttle_duration_sec=5.0)
            return
        cloud_stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        matched = self.segmentations.get(cloud_stamp_ns)
        if matched is None:
            self.pending_clouds[cloud_stamp_ns] = message
            while len(self.pending_clouds) > 8:
                del self.pending_clouds[next(iter(self.pending_clouds))]
            return
        self._process_cloud(message, matched[0])

    def _process_cloud(
        self, message: PointCloud2, segmentation: SegmentationResult
    ) -> None:
        """Fuse an already timestamp-matched image prediction and cloud."""

        start = time.perf_counter()
        try:
            points_source = pointcloud_message_to_xyz(message)
            camera_frame = self.camera_info.header.frame_id
            source_to_camera = self._lookup_matrix(
                camera_frame,
                message.header.frame_id,
                message.header.stamp,
            )
            source_to_base = self._lookup_matrix(
                self.base_frame,
                message.header.frame_id,
                message.header.stamp,
            )
            points_camera = transform_points(points_source, source_to_camera)
            points_base = transform_points(points_source, source_to_base)
            matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
            # The extracted A2D2 points and calibration use +x forward,
            # +y left, and +z up rather than the ROS optical convention.
            projection = project_camera_points(
                points_camera,
                matrix,
                (self.camera_info.width, self.camera_info.height),
            )
            painted = paint_points(
                points_camera,
                points_base,
                projection,
                segmentation,
            )
            costmap = build_semantic_costmap(
                painted,
                self.costmap_config,
                raw_points_vehicle=points_base,
                raytrace_origin_vehicle=source_to_base[:3, 3],
            )
        except (TransformException, ValueError) as error:
            self.get_logger().warning(f"Point-cloud fusion failed: {error}")
            return

        self._publish_painted_points(painted, message.header.stamp)
        self._publish_costmap(costmap, message.header.stamp)
        self.processed_clouds += 1
        if self.processed_clouds % 20 == 0:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.get_logger().info(
                f"Fused {len(painted.class_ids)} points in {elapsed_ms:.1f} ms"
            )

    def _publish_mask(self, segmentation: SegmentationResult, header: Header) -> None:
        message = Image()
        message.header = header
        message.height, message.width = segmentation.class_ids.shape
        message.encoding = "mono8"
        message.is_bigendian = False
        message.step = message.width
        message.data = segmentation.class_ids.tobytes()
        self.mask_publisher.publish(message)

    def _publish_painted_points(self, painted, stamp) -> None:
        header = Header(stamp=stamp, frame_id=self.base_frame)
        fields = [
            PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(
                ("x", "y", "z", "class_id", "confidence", "cost")
            )
        ]
        records = np.column_stack(
            (
                painted.points_vehicle.astype(np.float32),
                painted.class_ids.astype(np.float32),
                painted.confidence,
                painted.costs.astype(np.float32),
            )
        )
        self.painted_publisher.publish(
            point_cloud2.create_cloud(header, fields, records.tolist())
        )

    def _publish_costmap(self, costmap, stamp) -> None:
        message = OccupancyGrid()
        message.header = Header(stamp=stamp, frame_id=self.base_frame)
        message.info.resolution = costmap.config.resolution
        message.info.width = costmap.config.width
        message.info.height = costmap.config.height
        message.info.origin.position.x = costmap.config.x_min
        message.info.origin.position.y = costmap.config.y_min
        message.info.origin.orientation.w = 1.0
        occupancy = np.full(costmap.costs.shape, -1, dtype=np.int8)
        known = costmap.costs != 255
        occupancy[known] = np.rint(
            costmap.costs[known].astype(np.float32) * 100.0 / 254.0
        ).astype(np.int8)
        message.data = occupancy.ravel().tolist()
        self.costmap_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticCostmapNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
