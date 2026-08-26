"""Numpy geometry used by frame selection and depth backprojection."""

import cv2
import numpy as np
from scipy.spatial import cKDTree


def point_cloud_scales(points, fallback=0.03):
    """Estimate isotropic splat scales from the three nearest neighbors."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return np.full(len(points), fallback, dtype=np.float32)
    neighbor_count = min(4, len(points))
    distances, _ = cKDTree(points).query(points, k=neighbor_count, workers=-1)
    neighbors = distances[:, 1:]
    scales = np.sqrt(np.mean(np.square(neighbors), axis=1))
    scales[~np.isfinite(scales)] = fallback
    return scales.astype(np.float32)


def backproject(rgb, metric_depth, valid, intrinsics, transform, pixel_stride=2, voxel_size=0.03, max_points=50000):
    """Create a filtered world-space RGB point cloud from a depth map."""
    rows, columns = np.nonzero(valid)
    keep = (rows % pixel_stride == 0) & (columns % pixel_stride == 0)
    rows, columns = rows[keep], columns[keep]
    if not len(rows):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty(0, np.float32)
    z = metric_depth[rows, columns]
    fx, fy = intrinsics['fx'], intrinsics['fy']
    x = (columns - intrinsics['cx']) * z / fx
    y = (rows - intrinsics['cy']) * z / fy
    camera_points = np.column_stack((x, y, z))
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    translation = np.asarray(transform, dtype=np.float64)[:3, 3]
    points = camera_points @ rotation.T + translation
    colors = rgb[rows, columns].astype(np.float32) / 255.0
    scales = z.astype(np.float32) / float((fx + fy) / 2.0)
    if voxel_size > 0:
        cells = np.floor(points / voxel_size).astype(np.int64)
        _, unique = np.unique(cells, axis=0, return_index=True)
        points, colors, scales = points[unique], colors[unique], scales[unique]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points, colors, scales = points[indices], colors[indices], scales[indices]
    return points.astype(np.float32), colors.astype(np.float32), scales


def pose_distance(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    translation = np.linalg.norm(first[:3, 3] - second[:3, 3])
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(translation), float(np.degrees(np.arccos(cosine)))


def projected_overlap(depth, valid, intrinsics, current_transform, target_transform, samples=1600, border=20):
    """Estimate how much of current valid depth projects into target view."""
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return 0.0
    indices = np.linspace(0, len(rows) - 1, min(samples, len(rows))).astype(int)
    rows, columns = rows[indices], columns[indices]
    z = depth[rows, columns]
    x = (columns - intrinsics['cx']) * z / intrinsics['fx']
    y = (rows - intrinsics['cy']) * z / intrinsics['fy']
    points = np.column_stack((x, y, z, np.ones_like(z)))
    world = (np.asarray(current_transform) @ points.T).T
    target = (np.linalg.inv(np.asarray(target_transform)) @ world.T).T[:, :3]
    positive = target[:, 2] > 0
    u = intrinsics['fx'] * target[:, 0] / np.maximum(target[:, 2], 1e-8) + intrinsics['cx']
    v = intrinsics['fy'] * target[:, 1] / np.maximum(target[:, 2], 1e-8) + intrinsics['cy']
    height, width = depth.shape
    inside = positive & (u >= border) & (u < width - border)
    inside &= (v >= border) & (v < height - border)
    return float(inside.mean())


def resize_depth_preserving_validity(inverse_depth, valid, resolution):
    """Resize continuous depth without allowing invalid zeros to reappear."""
    resized_depth = cv2.resize(inverse_depth, resolution, interpolation=cv2.INTER_LINEAR)
    resized_valid = cv2.resize(valid.astype(np.uint8), resolution, interpolation=cv2.INTER_NEAREST).astype(bool)
    resized_valid &= np.isfinite(resized_depth) & (resized_depth > 0)
    resized_depth[~resized_valid] = 0
    return resized_depth, resized_valid


def reliable_depth_mask(metric_depth, valid, erosion_pixels=1, relative_edge_threshold=0.08):
    """Reject invalid neighborhoods and strong depth discontinuities."""
    safe = np.asarray(valid, dtype=bool).copy()
    if erosion_pixels > 0:
        size = 2 * erosion_pixels + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        safe &= cv2.erode(safe.astype(np.uint8), kernel, iterations=1).astype(bool)
    if relative_edge_threshold <= 0:
        return safe
    edge = np.zeros_like(safe)
    depth = np.asarray(metric_depth)
    for row_offset, column_offset in ((0, 1), (1, 0)):
        first = depth[: depth.shape[0] - row_offset or None, : depth.shape[1] - column_offset or None]
        second = depth[row_offset:, column_offset:]
        first_valid = valid[: valid.shape[0] - row_offset or None, : valid.shape[1] - column_offset or None]
        second_valid = valid[row_offset:, column_offset:]
        discontinuity = first_valid & second_valid
        discontinuity &= np.abs(first - second) > (relative_edge_threshold * np.minimum(first, second))
        edge[: edge.shape[0] - row_offset or None, : edge.shape[1] - column_offset or None] |= discontinuity
        edge[row_offset:, column_offset:] |= discontinuity
    return safe & ~edge


def should_add_keyframe(
    last_packet,
    packet,
    overlap,
    median_depth,
    max_gap=5,
    overlap_threshold=0.65,
    translation_depth_ratio=0.05,
    rotation_threshold_deg=10.0,
    min_gap=0,
):
    if last_packet is None:
        return True
    gap = packet.sequence_id - last_packet.sequence_id
    if gap < min_gap:
        return False
    if gap >= max_gap:
        return True
    translation, rotation = pose_distance(last_packet.T_world_camera, packet.T_world_camera)
    return bool(overlap < overlap_threshold or translation > translation_depth_ratio * max(median_depth, 1e-6) or rotation > rotation_threshold_deg)
