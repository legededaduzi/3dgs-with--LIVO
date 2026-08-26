"""Launch frame archiving and the independently configured GPU backend."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory('gs_slam'))
    session = LaunchConfiguration('session')
    frontend_configuration = LaunchConfiguration('frontend_config')
    backend_configuration = LaunchConfiguration('backend_config')
    backend_python = LaunchConfiguration('backend_python')
    backend_module_path = LaunchConfiguration('backend_module_path')
    preview = LaunchConfiguration('preview')
    preview_depth_min = LaunchConfiguration('preview_depth_min')
    preview_depth_max = LaunchConfiguration('preview_depth_max')
    return LaunchDescription(
        [
            DeclareLaunchArgument('session', default_value='/home/ubuntu/GS-SLAM/data', description='Shared frontend/backend session directory'),
            DeclareLaunchArgument(
                'frontend_config',
                default_value=str(package_share / 'config' / 'stereo_camera.yaml'),
                description='ROS topics, calibration, sampling, and archive layout',
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
            DeclareLaunchArgument(
                'preview', default_value='false', choices=['true', 'false'], description='Show the low-overhead fixed-camera mapping preview'
            ),
            DeclareLaunchArgument('preview_depth_min', default_value='0.2', description='Nearest metric depth shown by the preview'),
            DeclareLaunchArgument('preview_depth_max', default_value='5.0', description='Farthest metric depth shown by the preview'),
            Node(
                package='gs_slam',
                executable='map_generator',
                name='frame_archiver',
                output='screen',
                parameters=[frontend_configuration, {'output.directory': session}],
            ),
            ExecuteProcess(
                cmd=[
                    backend_python,
                    '-m',
                    'gs_slam_backend.runner',
                    'live',
                    '--session',
                    session,
                    '--config',
                    backend_configuration,
                    '--preview',
                    preview,
                    '--preview-depth-min',
                    preview_depth_min,
                    '--preview-depth-max',
                    preview_depth_max,
                ],
                additional_env={'PYTHONPATH': backend_module_path},
                output='screen',
                # Final PLY serialization can take several seconds for a
                # million-Gaussian map. Let the backend finish its atomic save.
                sigterm_timeout='300',
                sigkill_timeout='60',
            ),
        ]
    )
