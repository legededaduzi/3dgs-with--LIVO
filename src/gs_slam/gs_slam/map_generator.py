"""Downsample the ROS stream and archive backend-ready RGB-D frames."""

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from gs_slam.frame_archive import FrameArchive
from gs_slam.utils import rotation_matrix
from gs_slam.utils import stamp_ns


class FrameArchiver(Node):

    def __init__(self):
        super().__init__('frame_archiver')
        defaults = {
            'left_image_topic': '/camera/infra1/image_raw',
            'right_image_topic': '/camera/infra2/image_raw',
            'color_image_topic': '/camera/color/image_raw',
            'odometry_topic': '/odometry',
            'camera_intrinsics.fx': 320.25493621826172,
            'camera_intrinsics.fy': 320.25492668151855,
            'camera_intrinsics.cx': 320.0,
            'camera_intrinsics.cy': 240.0,
            'camera_extrinsics.baseline': 0.08,
            'camera_extrinsics.left.translation': [0.1, 0.04, 0.0],
            'camera_extrinsics.left.rotation_xyzw': [-0.5, 0.5, -0.5, 0.5],
            'camera_extrinsics.color.translation': [0.1, 0.05, 0.0],
            'camera_extrinsics.color.rotation_xyzw': [-0.5, 0.5, -0.5, 0.5],
            'processing.rate_hz': 1.0,
            'processing.stereo_tolerance_ms': 5.0,
            'processing.pair_tolerance_ms': 20.0,
            'processing.sync_queue_size': 30,
            'output.directory': './data',
            'output.image_directory': 'images',
            'output.depth_directory': 'depth',
            'output.sparse_directory': 'sparse/0',
            'output.manifest_directory': 'manifests',
            'stereo_matching.min_disparity': 0,
            'stereo_matching.num_disparities': 128,
            'stereo_matching.block_size': 5,
            'stereo_matching.uniqueness_ratio': 10,
            'stereo_matching.speckle_window_size': 100,
            'stereo_matching.speckle_range': 2,
            'stereo_matching.disp12_max_diff': 1,
            'depth_confidence.disparity_sigma_px': 0.5,
            'depth_confidence.max_lr_error_px': 1.0,
            'depth_confidence.max_relative_uncertainty': 0.15,
        }
        self.p = {name: self.declare_parameter(name, default).value for name, default in defaults.items()}
        self.period_ns = int(1e9 / self.p['processing.rate_hz'])
        self.stereo_tolerance_ns = int(self.p['processing.stereo_tolerance_ms'] * 1e6)

        self.bridge = CvBridge()
        self.processor = FrameArchive(self.p)
        self.next_stamp = None
        self.frame_id = None

        specs = [
            (Image, self.p['left_image_topic']),
            (Image, self.p['right_image_topic']),
            (Image, self.p['color_image_topic']),
            (Odometry, self.p['odometry_topic']),
        ]
        self.subscribers = [Subscriber(self, message_type, topic, qos_profile=qos_profile_sensor_data) for message_type, topic in specs]
        self.synchronizer = ApproximateTimeSynchronizer(
            self.subscribers, self.p['processing.sync_queue_size'], self.p['processing.pair_tolerance_ms'] / 1000.0
        )
        self.synchronizer.registerCallback(self.archive_frame)
        self.get_logger().info(
            'Archiving at %.1f Hz; RGB: %s; depth: %s; metadata: %s'
            % (self.p['processing.rate_hz'], self.processor.image_directory, self.processor.depth_directory, self.processor.sparse_directory)
        )

    def archive_frame(self, left_message, right_message, color_message, odometry):

        current_stamp = stamp_ns(left_message)
        stereo_gap = abs(current_stamp - stamp_ns(right_message))
        if stereo_gap > self.stereo_tolerance_ns:
            return
        if self.next_stamp is not None and current_stamp < self.next_stamp:
            return
        if self.frame_id not in (None, odometry.header.frame_id):
            return
        self.frame_id = odometry.header.frame_id

        left = self.bridge.imgmsg_to_cv2(left_message, 'mono8')
        right = self.bridge.imgmsg_to_cv2(right_message, 'mono8')
        color_image = self.bridge.imgmsg_to_cv2(color_message, 'bgr8')

        pose = odometry.pose.pose
        world_rotation = rotation_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        world_translation = np.array([pose.position.x, pose.position.y, pose.position.z])

        try:
            sample_count = self.processor.archive_frame(
                left, right, color_image, world_rotation, world_translation, timestamp_ns=current_stamp, world_frame=self.frame_id or ''
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.get_logger().error('Could not process synchronized frame: %s' % error)
            return
        if sample_count is None:
            return

        self.next_stamp = current_stamp + self.period_ns
        self.get_logger().info('Archived frame %d (%d valid depth samples)' % (self.processor.frame_count, sample_count))

    def save(self):
        """Flush archive metadata."""
        try:
            result = self.processor.save()
        except OSError as error:
            self.get_logger().error('Could not save mapping output: %s' % error)
            return
        if result is None:
            return

        sample_count, frame_count = result
        self.get_logger().info('Saved %d frames (%d valid depth samples) to %s' % (frame_count, sample_count, self.processor.output_directory))


def main(args=None):
    """Run mapping and save COLMAP data when the process exits."""
    rclpy.init(args=args)
    node = FrameArchiver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.save()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
