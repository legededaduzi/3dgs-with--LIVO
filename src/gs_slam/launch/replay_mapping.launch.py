"""Replay an archived RGB-D session through the Gaussian backend."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    """Configure and start one finite backend replay process."""
    package_share = Path(get_package_share_directory('gs_slam'))
    source = LaunchConfiguration('source')
    output = LaunchConfiguration('output')
    backend_configuration = LaunchConfiguration('backend_config')
    backend_python = LaunchConfiguration('backend_python')
    backend_module_path = LaunchConfiguration('backend_module_path')
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'source', default_value='/home/ubuntu/GS-SLAM/data/session_01', description='Archived session containing manifests or COLMAP data'
            ),
            DeclareLaunchArgument(
                'output',
                default_value=PathJoinSubstitution([source, 'replay_output']),
                description='Directory for replay status, checkpoints, and PLY',
            ),
            DeclareLaunchArgument(
                'backend_config',
                default_value=str(package_share / 'config' / 'online_backend.json'),
                description='All Gaussian mapping switches and parameters',
            ),
            DeclareLaunchArgument(
                'backend_python',
                default_value='/home/ubuntu/miniconda3/envs/3dgs/bin/python',
                description='Python interpreter containing Torch and rasterizer',
            ),
            DeclareLaunchArgument(
                'backend_module_path',
                default_value='/home/ubuntu/GS-SLAM/src/gs_slam_backend',
                description='Directory containing the gs_slam_backend package',
            ),
            ExecuteProcess(
                cmd=[
                    backend_python,
                    '-m',
                    'gs_slam_backend.runner',
                    'replay',
                    '--source',
                    source,
                    '--output',
                    output,
                    '--config',
                    backend_configuration,
                ],
                additional_env={'PYTHONPATH': backend_module_path},
                output='screen',
                sigterm_timeout='300',
                sigkill_timeout='60',
            ),
        ]
    )
