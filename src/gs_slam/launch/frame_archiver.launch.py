"""Launch the standalone ROS frame archiver."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Load frontend configuration and start frame archiving."""
    package_share = Path(get_package_share_directory('gs_slam'))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'frontend_config',
                default_value=str(package_share / 'config' / 'stereo_camera.yaml'),
                description='ROS topics, calibration, sampling, and archive layout',
            ),
            Node(
                package='gs_slam',
                executable='map_generator',
                name='frame_archiver',
                output='screen',
                parameters=[LaunchConfiguration('frontend_config')],
            ),
        ]
    )
