from setuptools import find_packages, setup

package_name = 'gs_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/stereo_camera.yaml', 'config/online_backend.json']),
        (
            'share/' + package_name + '/launch',
            ['launch/frame_archiver.launch.py', 'launch/online_mapping.launch.py', 'launch/replay_mapping.launch.py'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='b',
    maintainer_email='2861086514@qq.com',
    description='ROS stereo frame archiver for the GS-SLAM backend',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': ['map_generator = gs_slam.map_generator:main']},
)
