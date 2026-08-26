"""Process stereo data and export a COLMAP-compatible dataset."""

import json
from pathlib import Path
import struct

import cv2
import numpy as np
from scipy.spatial import cKDTree

from gs_slam.utils import colmap_image_pose
from gs_slam.utils import rotation_matrix


def atomic_binary_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + '.tmp')


def write_colmap_cameras(path, width, height, intrinsics):
    """Write one COLMAP PINHOLE camera to ``cameras.bin``."""
    temporary = atomic_binary_writer(path)
    fx, fy, cx, cy = intrinsics
    has_camera = width > 0 and height > 0
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<Q', int(has_camera)))
        if has_camera:
            # CAMERA_ID, MODEL_ID (PINHOLE), WIDTH, HEIGHT, PARAMS[]
            stream.write(struct.pack('<iiQQdddd', 1, 1, width, height, fx, fy, cx, cy))
    temporary.replace(path)


def write_colmap_images(path, images):
    """Write registered camera poses without feature observations."""
    temporary = atomic_binary_writer(path)
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<Q', len(images)))
        for image_id, quaternion, translation, name in images:
            stream.write(
                struct.pack('<idddddddi', image_id, *np.asarray(quaternion, dtype=np.float64), *np.asarray(translation, dtype=np.float64), 1)
            )
            stream.write(name.encode('utf-8') + b'\0')
            stream.write(struct.pack('<Q', 0))
    temporary.replace(path)


def write_colmap_points3d(path, points, colors):
    """Write colored map points without feature tracks."""
    temporary = atomic_binary_writer(path)
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        point_colors = enumerate(zip(points, colors), start=1)
        for point_id, (point, color) in point_colors:
            stream.write(struct.pack('<QdddBBBdQ', point_id, *np.asarray(point, dtype=np.float64), *np.asarray(color, dtype=np.uint8), 0.0, 0))
    temporary.replace(path)


def write_png(path, image):
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError('OpenCV could not encode the frame as PNG')
    temporary = atomic_binary_writer(path)
    with temporary.open('wb') as stream:
        stream.write(encoded.tobytes())
    temporary.replace(path)


def write_depth_params(path, parameters):
    """Write Graphdeco 3DGS inverse-depth alignment parameters."""
    temporary = atomic_binary_writer(path)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(parameters, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def write_frame_manifest(path, packet):
    """Atomically publish one complete online-backend frame packet."""
    temporary = atomic_binary_writer(path)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(packet, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def frame_filename(frame_index):
    """Return the shared zero-based RGB and depth filename."""
    return '%d.png' % frame_index


def render_depth_image(camera_points, image_shape, intrinsics, source_rotation, source_translation, target_rotation, target_translation):
    """Project source-camera points into a metric target-camera Z map."""
    height, width = image_shape
    depth_image = np.zeros((height, width), dtype=np.float32)
    if not len(camera_points):
        return depth_image

    body_points = camera_points @ source_rotation.T + source_translation
    target_points = (body_points - target_translation) @ target_rotation
    z = target_points[:, 2]
    visible = np.isfinite(target_points).all(axis=1) & (z > 0)
    if not np.any(visible):
        return depth_image

    fx, fy, cx, cy = intrinsics
    target_points = target_points[visible]
    z = target_points[:, 2]
    u = np.rint(fx * target_points[:, 0] / z + cx).astype(np.int64)
    v = np.rint(fy * target_points[:, 1] / z + cy).astype(np.int64)
    visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(visible):
        return depth_image

    u, v, z = u[visible], v[visible], z[visible]
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, (v, u), z)
    valid = np.isfinite(depth_buffer)
    depth_image[valid] = depth_buffer[valid]
    return depth_image


def disparity_confidence(left_disparity, right_disparity, disparity_sigma_px, max_lr_error_px, max_relative_uncertainty):
    """Estimate confidence from LR consistency and disparity uncertainty."""
    if left_disparity.shape != right_disparity.shape:
        raise ValueError('Left and right disparity shapes do not match')
    if disparity_sigma_px <= 0:
        raise ValueError('Disparity sigma must be greater than zero')
    if max_lr_error_px <= 0 or max_relative_uncertainty <= 0:
        raise ValueError('Confidence thresholds must be greater than zero')

    height, width = left_disparity.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    right_x = grid_x - left_disparity
    right_at_left = cv2.remap(right_disparity, right_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    valid = np.isfinite(left_disparity) & (left_disparity > 0)
    valid &= np.isfinite(right_at_left) & (right_at_left < 0)
    valid &= (right_x >= 0) & (right_x < width)
    lr_error = np.abs(left_disparity + right_at_left)
    valid &= lr_error <= max_lr_error_px

    estimated_sigma = np.sqrt(disparity_sigma_px**2 + lr_error**2)
    relative_uncertainty = np.full(left_disparity.shape, np.inf, dtype=np.float32)
    relative_uncertainty[valid] = estimated_sigma[valid] / left_disparity[valid]
    confidence = np.clip(1.0 - relative_uncertainty / max_relative_uncertainty, 0.0, 1.0).astype(np.float32)
    confidence[~valid] = 0.0
    return confidence


def encode_graphdeco_inverse_depth(depth_image):
    """Encode metric Z as uint16 inverse depth while preserving invalid 0."""
    valid = np.isfinite(depth_image) & (depth_image > 0)
    if not np.any(valid):
        raise ValueError('Cannot encode an inverse-depth map without depth')

    inverse_depth = np.zeros(depth_image.shape, dtype=np.float32)
    inverse_depth[valid] = 1.0 / depth_image[valid]

    maximum_inverse_depth = float(np.max(inverse_depth[valid]))
    normalized = np.clip(inverse_depth / maximum_inverse_depth, 0.0, 1.0)
    encoded = np.rint(normalized * np.iinfo(np.uint16).max).astype(np.uint16)

    # The official loader first divides the PNG by 2**16 and then applies
    # this JSON scale and offset. This scale recovers inverse depth in 1/m.
    alignment_scale = maximum_inverse_depth * float(2**16) / np.iinfo(np.uint16).max
    return encoded, alignment_scale


def downsample(points, colors, size):
    if not len(points) or size <= 0:
        return points.astype(np.float32), colors.astype(np.uint8)

    _, group = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_inverse=True)

    count = np.bincount(group)

    points = np.column_stack([np.bincount(group, weights=points[:, axis]) / count for axis in range(3)]).astype(np.float32)

    colors = np.column_stack([np.bincount(group, weights=colors[:, channel]) / count for channel in range(3)])
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    # 简单半径降噪：每个点在 2 个体素范围内至少需要另一个邻点。
    if len(points) > 1:
        neighbor_count = cKDTree(points).query_ball_point(points, r=2.0 * size, return_length=True, workers=-1)
        keep = neighbor_count >= 2  # 邻域计数包含查询点自身
        points = points[keep]
        colors = colors[keep]

    return points, colors


class FrameArchive:
    """
    Encode and atomically archive synchronized RGB-D frame packets.

    The historical class name remains import-compatible. It no longer owns a
    persistent point-cloud map; Gaussian map state belongs to the backend.
    """

    def __init__(self, parameters):
        self.p = parameters
        output_directory = Path(self.p['output.directory'])
        output_directory = output_directory.expanduser().resolve()
        self.image_directory = output_directory / self.p['output.image_directory']
        self.depth_directory = output_directory / self.p['output.depth_directory']
        self.sparse_directory = output_directory / self.p['output.sparse_directory']
        self.manifest_directory = output_directory / self.p.get('output.manifest_directory', 'manifests')
        self.image_directory.mkdir(parents=True, exist_ok=True)
        self.depth_directory.mkdir(parents=True, exist_ok=True)
        self.sparse_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = output_directory

        block = self.p['stereo_matching.block_size']
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.p['stereo_matching.min_disparity'],
            numDisparities=self.p['stereo_matching.num_disparities'],
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            disp12MaxDiff=self.p['stereo_matching.disp12_max_diff'],
            uniquenessRatio=self.p['stereo_matching.uniqueness_ratio'],
            speckleWindowSize=self.p['stereo_matching.speckle_window_size'],
            speckleRange=self.p['stereo_matching.speckle_range'],
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)

        fx, fy = self.p['camera_intrinsics.fx'], self.p['camera_intrinsics.fy']
        cx, cy = self.p['camera_intrinsics.cx'], self.p['camera_intrinsics.cy']
        baseline = self.p['camera_extrinsics.baseline']

        self.reprojection = np.float32([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, fx], [0, 0, 1 / baseline, 0]])

        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        self.left_rotation = rotation_matrix(self.p['camera_extrinsics.left.rotation_xyzw'])
        self.left_translation = np.asarray(self.p['camera_extrinsics.left.translation'])
        self.color_rotation = rotation_matrix(self.p['camera_extrinsics.color.rotation_xyzw'])
        self.color_translation = np.asarray(self.p['camera_extrinsics.color.translation'])

        self.frame_count = 0
        self.valid_depth_samples = 0
        self.colmap_images = []
        self.depth_params = {}
        self.image_size = None
        self.closed = False

    def archive_frame(self, left, right, color_image, world_rotation, world_translation, timestamp_ns=0, world_frame=''):

        image_size = (color_image.shape[1], color_image.shape[0])
        if self.image_size not in (None, image_size):
            raise ValueError('RGB frame size changed to %dx%d (expected %dx%d)' % (*image_size, *self.image_size))
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        right_disparity = self.right_matcher.compute(right, left).astype(np.float32) / 16.0
        camera_points = cv2.reprojectImageTo3D(disparity, self.reprojection)

        depth = camera_points[:, :, 2]
        confidence = disparity_confidence(
            disparity,
            right_disparity,
            self.p['depth_confidence.disparity_sigma_px'],
            self.p['depth_confidence.max_lr_error_px'],
            self.p['depth_confidence.max_relative_uncertainty'],
        )
        reliable = confidence > 0
        reliable &= np.isfinite(camera_points).all(axis=2)
        reliable &= depth > 0
        depth_image = render_depth_image(
            camera_points[reliable],
            color_image.shape[:2],
            (self.fx, self.fy, self.cx, self.cy),
            self.left_rotation,
            self.left_translation,
            self.color_rotation,
            self.color_translation,
        )
        inverse_depth, depth_scale = encode_graphdeco_inverse_depth(depth_image)

        sample_count = int(np.count_nonzero(depth_image))
        if not sample_count:
            return None
        image_id = self.frame_count + 1
        image_name = frame_filename(self.frame_count)

        write_png(self.image_directory / image_name, color_image)
        write_png(self.depth_directory / image_name, inverse_depth)
        quaternion, translation = colmap_image_pose(world_rotation, world_translation, self.color_rotation, self.color_translation)
        self.colmap_images.append((image_id, quaternion, translation, image_name))
        self.depth_params[str(self.frame_count)] = {'scale': depth_scale, 'offset': 0.0}
        world_from_camera_rotation = world_rotation @ self.color_rotation
        world_from_camera_translation = world_rotation @ self.color_translation + world_translation
        world_from_camera = np.eye(4, dtype=np.float64)
        world_from_camera[:3, :3] = world_from_camera_rotation
        world_from_camera[:3, 3] = world_from_camera_translation
        packet = {
            'schema_version': 1,
            'sequence_id': self.frame_count,
            'timestamp_ns': int(timestamp_ns),
            'frame_name': image_name,
            'world_frame': world_frame,
            'camera_frame': 'camera_color_optical_frame',
            'rgb_path': str((self.image_directory / image_name).relative_to(self.output_directory)),
            'inverse_depth_path': str((self.depth_directory / image_name).relative_to(self.output_directory)),
            'inverse_depth_scale': float(depth_scale),
            'inverse_depth_offset': 0.0,
            'width': int(image_size[0]),
            'height': int(image_size[1]),
            'intrinsics': {'fx': float(self.fx), 'fy': float(self.fy), 'cx': float(self.cx), 'cy': float(self.cy)},
            'T_world_camera': world_from_camera.tolist(),
        }
        write_frame_manifest(self.manifest_directory / ('%09d.json' % self.frame_count), packet)
        self.image_size = image_size
        self.valid_depth_samples += sample_count
        self.frame_count += 1
        return sample_count

    # Compatibility for callers using the old mapping-oriented method name.
    process_single_colored_point_cloud = archive_frame

    def save(self):
        """Save COLMAP camera/pose metadata without frontend map state."""
        if self.closed:
            return None
        self.closed = True
        width, height = self.image_size or (0, 0)
        write_colmap_cameras(self.sparse_directory / 'cameras.bin', width, height, (self.fx, self.fy, self.cx, self.cy))
        write_colmap_images(self.sparse_directory / 'images.bin', self.colmap_images)
        write_colmap_points3d(self.sparse_directory / 'points3D.bin', np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
        write_depth_params(self.sparse_directory / 'depth_params.json', self.depth_params)
        return self.valid_depth_samples, self.frame_count


# Compatibility for downstream code that imported the historical class.
ColoredPointCloudProcessor = FrameArchive
