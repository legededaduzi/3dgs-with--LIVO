"""
Public frontend archive API.

Storage helpers remain in the legacy module so existing dataset tooling keeps
working while the ROS node depends only on the narrower archive abstraction.
"""

from gs_slam.colored_pointcloud_processor import FrameArchive

__all__ = ['FrameArchive']
