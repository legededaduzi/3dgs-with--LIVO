# Copyright (C) 2023, Inria, GRAPHDECO research group.
# All rights reserved. Research/evaluation use under ../LICENSE.md.
"""GPU camera object adapted from Graphdeco Gaussian Splatting."""

import math

import numpy as np
import torch


def projection_matrix(znear, zfar, fov_x, fov_y, device):
    tan_y, tan_x = math.tan(fov_y / 2), math.tan(fov_x / 2)
    top, right = tan_y * znear, tan_x * znear
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = znear / right
    matrix[1, 1] = znear / top
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


class GaussianCamera:
    """A fixed-pose camera ready for the CUDA rasterizer."""

    def __init__(self, packet, rgb, inverse_depth, valid, device='cuda:0'):
        self.packet = packet
        self.sequence_id = packet.sequence_id
        self.image_name = packet.frame_name
        self.image_width = packet.width
        self.image_height = packet.height
        self.device = torch.device(device)
        self.original_image = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).to(self.device)
        self.inverse_depth = torch.from_numpy(inverse_depth[None].astype(np.float32)).to(self.device)
        self.depth_mask = torch.from_numpy(valid[None].astype(np.float32)).to(self.device)
        transform = np.asarray(packet.T_world_camera, dtype=np.float32)
        world_view = np.linalg.inv(transform)
        self.world_view_transform = torch.from_numpy(world_view).to(self.device).transpose(0, 1)
        fx, fy = packet.intrinsics['fx'], packet.intrinsics['fy']
        self.fov_x = 2.0 * math.atan(packet.width / (2.0 * fx))
        self.fov_y = 2.0 * math.atan(packet.height / (2.0 * fy))
        projection = projection_matrix(0.01, 100.0, self.fov_x, self.fov_y, self.device).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        self.camera_center = torch.from_numpy(transform[:3, 3]).to(self.device)
