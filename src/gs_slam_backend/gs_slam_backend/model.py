# Copyright (C) 2023, Inria, GRAPHDECO research group.
# All rights reserved. Research/evaluation use under ../LICENSE.md.
"""Mutable Gaussian map adapted from Graphdeco Gaussian Splatting."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    SparseGaussianAdam = None


SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def inverse_sigmoid(value):
    return torch.log(value / (1.0 - value))


def quaternion_to_matrix(quaternion):
    """Convert normalized scalar-first quaternions to rotation matrices."""
    quaternion = torch.nn.functional.normalize(quaternion, dim=1)
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


class GaussianMap(nn.Module):
    """Graphdeco-compatible Gaussians with safe online append/prune."""

    parameter_names = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
    parameter_attributes = {
        'xyz': '_xyz',
        'f_dc': '_features_dc',
        'f_rest': '_features_rest',
        'opacity': '_opacity',
        'scaling': '_scaling',
        'rotation': '_rotation',
    }

    def __init__(self, sh_degree=3, device='cuda:0', initial_opacity=0.9):
        super().__init__()
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.device = torch.device(device)
        self.initial_opacity = float(initial_opacity)
        self._xyz = nn.Parameter(torch.empty((0, 3), device=self.device))
        self._features_dc = nn.Parameter(torch.empty((0, 1, 3), device=self.device))
        self._features_rest = nn.Parameter(torch.empty((0, (sh_degree + 1) ** 2 - 1, 3), device=self.device))
        self._opacity = nn.Parameter(torch.empty((0, 1), device=self.device))
        self._scaling = nn.Parameter(torch.empty((0, 3), device=self.device))
        self._rotation = nn.Parameter(torch.empty((0, 4), device=self.device))
        self.optimizer = None
        self.max_radii2d = torch.empty(0, device=self.device)
        self.birth_keyframe = torch.empty(0, dtype=torch.long, device=self.device)
        self.last_seen_keyframe = torch.empty(0, dtype=torch.long, device=self.device)
        self.depth_inconsistency = torch.empty(0, dtype=torch.long, device=self.device)
        self.xyz_gradient_accum = torch.empty(0, device=self.device)
        self.gradient_denom = torch.empty(0, device=self.device)

    @property
    def xyz(self):
        return self._xyz

    @property
    def features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def scaling(self):
        return torch.exp(self._scaling)

    @property
    def rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    def render_subset(self, indices):
        """Return differentiable render parameters for selected rows only."""
        features_dc = self._features_dc[indices]
        features_rest = self._features_rest[indices]
        return {
            'xyz': self._xyz[indices],
            'features': torch.cat((features_dc, features_rest), dim=1),
            'features_dc': features_dc,
            'features_rest': features_rest,
            'opacity': torch.sigmoid(self._opacity[indices]),
            'scaling': torch.exp(self._scaling[indices]),
            'rotation': torch.nn.functional.normalize(self._rotation[indices]),
        }

    def initialize(self, points, colors, scales, keyframe=0, opacity=None):
        if not len(points):
            raise ValueError('Cannot initialize an empty Gaussian map')
        tensors = self._new_tensors(points, colors, scales, opacity=opacity)
        for name, tensor in tensors.items():
            setattr(self, self.parameter_attributes[name], nn.Parameter(tensor.requires_grad_(True)))
        count = len(points)
        self.max_radii2d = torch.zeros(count, device=self.device)
        self.birth_keyframe = torch.full((count,), keyframe, dtype=torch.long, device=self.device)
        self.last_seen_keyframe = self.birth_keyframe.clone()
        self.depth_inconsistency = torch.zeros(count, dtype=torch.long, device=self.device)
        self.xyz_gradient_accum = torch.zeros(count, device=self.device)
        self.gradient_denom = torch.zeros(count, device=self.device)

    def _new_tensors(self, points, colors, scales, opacity=None):
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        scales = torch.as_tensor(scales, dtype=torch.float32, device=self.device)
        scales = scales.clamp_min(1e-5)[:, None].repeat(1, 3).log()
        dc = rgb_to_sh(colors)[:, None, :]
        rest = torch.zeros((len(points), (self.max_sh_degree + 1) ** 2 - 1, 3), device=self.device)
        rotations = torch.zeros((len(points), 4), device=self.device)
        rotations[:, 0] = 1
        # A moderate initial opacity lets new depth points receive gradients
        # without making a single noisy observation immediately dominant.
        initial_opacity = self.initial_opacity if opacity is None else opacity
        opacity = inverse_sigmoid(torch.full((len(points), 1), float(initial_opacity), device=self.device))
        return dict(xyz=points, f_dc=dc, f_rest=rest, opacity=opacity, scaling=scales, rotation=rotations)

    def setup_optimizer(self, config):
        groups = [
            {'params': [self._xyz], 'lr': config.position_lr, 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': config.feature_lr, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': config.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self._opacity], 'lr': config.opacity_lr, 'name': 'opacity'},
            {'params': [self._scaling], 'lr': config.scaling_lr, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': config.rotation_lr, 'name': 'rotation'},
        ]
        if config.optimizer_type == 'sparse_adam':
            if SparseGaussianAdam is None:
                raise RuntimeError('SparseGaussianAdam is unavailable; install a ' 'diff-gaussian-rasterization build with sparse Adam support')
            if self.device.type != 'cuda':
                raise RuntimeError('SparseGaussianAdam requires a CUDA device')
            self.optimizer = SparseGaussianAdam(groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_position_learning_rate(self, step, final_lr, max_steps):
        """Apply Graphdeco's log-linear position learning-rate schedule."""
        if final_lr is None or max_steps <= 0:
            return
        group = next(value for value in self.optimizer.param_groups if value['name'] == 'xyz')
        initial_lr = group.get('initial_lr', group['lr'])
        group['initial_lr'] = initial_lr
        progress = min(max(float(step) / max_steps, 0.0), 1.0)
        group['lr'] = float(np.exp(np.log(initial_lr) * (1.0 - progress) + np.log(final_lr) * progress))

    def constrain_parameters(self, minimum_opacity=0.001, maximum_opacity=0.99, minimum_scale=0.001, maximum_scale=0.05, maximum_anisotropy=0.0):
        """Keep optimized splat attributes in physically useful ranges."""
        with torch.no_grad():
            minimum_dc = (-0.5) / SH_C0
            maximum_dc = 0.5 / SH_C0
            self._features_dc.clamp_(minimum_dc, maximum_dc)
            self._opacity.clamp_(float(inverse_sigmoid(torch.tensor(minimum_opacity))), float(inverse_sigmoid(torch.tensor(maximum_opacity))))
            self._scaling.clamp_(float(np.log(minimum_scale)), float(np.log(maximum_scale)))
            if maximum_anisotropy > 1.0:
                # Work in log-scale space and only shorten runaway axes.  This
                # preserves the smallest, depth-supported axis while preventing
                # weakly constrained online splats from becoming long needles.
                minimum_log_scale = self._scaling.min(dim=1, keepdim=True).values
                maximum_log_scale = minimum_log_scale + float(np.log(maximum_anisotropy))
                self._scaling.copy_(torch.minimum(self._scaling, maximum_log_scale))
            normalized = torch.nn.functional.normalize(self._rotation, dim=1)
            self._rotation.copy_(normalized)

    def _mutate_optimizer(self, extensions=None, keep=None):
        if self.optimizer is None:
            raise RuntimeError('Optimizer must be initialized before map mutation')
        updated = {}
        for group in self.optimizer.param_groups:
            name = group['name']
            old = group['params'][0]
            state = self.optimizer.state.pop(old, None)
            if extensions is not None:
                value = torch.cat((old.detach(), extensions[name].detach()), dim=0)
            else:
                value = old.detach()[keep]
            parameter = nn.Parameter(value.requires_grad_(True))
            group['params'][0] = parameter
            if state is not None:
                for state_name in ('exp_avg', 'exp_avg_sq'):
                    if extensions is not None:
                        state[state_name] = torch.cat((state[state_name], torch.zeros_like(extensions[name])), dim=0)
                    else:
                        state[state_name] = state[state_name][keep]
                self.optimizer.state[parameter] = state
            updated[name] = parameter
        self._xyz = updated['xyz']
        self._features_dc = updated['f_dc']
        self._features_rest = updated['f_rest']
        self._opacity = updated['opacity']
        self._scaling = updated['scaling']
        self._rotation = updated['rotation']

    def append(self, points, colors, scales, keyframe):
        if not len(points):
            return 0
        tensors = self._new_tensors(points, colors, scales)
        self._mutate_optimizer(extensions=tensors)
        count = len(points)
        self.max_radii2d = torch.cat((self.max_radii2d, torch.zeros(count, device=self.device)))
        born = torch.full((count,), keyframe, dtype=torch.long, device=self.device)
        self.birth_keyframe = torch.cat((self.birth_keyframe, born))
        self.last_seen_keyframe = torch.cat((self.last_seen_keyframe, born.clone()))
        self.depth_inconsistency = torch.cat((self.depth_inconsistency, torch.zeros(count, dtype=torch.long, device=self.device)))
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros(count, device=self.device)))
        self.gradient_denom = torch.cat((self.gradient_denom, torch.zeros(count, device=self.device)))
        return count

    def prune(self, prune_mask):
        prune_mask = prune_mask.to(self.device, dtype=torch.bool)
        if not prune_mask.any():
            return 0
        keep = ~prune_mask
        removed = int(prune_mask.sum())
        self._mutate_optimizer(keep=keep)
        self.max_radii2d = self.max_radii2d[keep]
        self.birth_keyframe = self.birth_keyframe[keep]
        self.last_seen_keyframe = self.last_seen_keyframe[keep]
        self.depth_inconsistency = self.depth_inconsistency[keep]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.gradient_denom = self.gradient_denom[keep]
        return removed

    def add_densification_stats(self, indices, viewspace_gradient):
        """Accumulate visible screen-space gradient magnitudes."""
        if not len(indices):
            return
        gradient = torch.linalg.vector_norm(viewspace_gradient[:, :2], dim=1)
        self.xyz_gradient_accum[indices] += gradient
        self.gradient_denom[indices] += 1

    def _append_internal(self, tensors, source_indices, keyframe):
        count = len(source_indices)
        if not count:
            return 0
        self._mutate_optimizer(extensions=tensors)
        self.max_radii2d = torch.cat((self.max_radii2d, torch.zeros(count, device=self.device)))
        born = self.birth_keyframe[source_indices].clone()
        seen = self.last_seen_keyframe[source_indices].clone()
        inconsistency = self.depth_inconsistency[source_indices].clone()
        if keyframe >= 0:
            born.fill_(keyframe)
            seen.fill_(keyframe)
            inconsistency.zero_()
        self.birth_keyframe = torch.cat((self.birth_keyframe, born))
        self.last_seen_keyframe = torch.cat((self.last_seen_keyframe, seen))
        self.depth_inconsistency = torch.cat((self.depth_inconsistency, inconsistency))
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros(count, device=self.device)))
        self.gradient_denom = torch.cat((self.gradient_denom, torch.zeros(count, device=self.device)))
        return count

    def densify_and_prune(self, grad_threshold, percent_dense, scene_extent, min_opacity, max_points, keyframe):
        """Clone small high-gradient splats and split large ones."""
        gradients = self.xyz_gradient_accum / self.gradient_denom.clamp_min(1)
        candidates = gradients >= grad_threshold
        candidate_indices = torch.where(candidates)[0]
        if max_points > 0 and len(candidate_indices) > max_points:
            values = gradients[candidate_indices]
            candidate_indices = candidate_indices[torch.topk(values, max_points, sorted=False).indices]
        if not len(candidate_indices):
            self.xyz_gradient_accum.zero_()
            self.gradient_denom.zero_()
            return 0, 0

        scale_limit = percent_dense * max(float(scene_extent), 1e-6)
        candidate_scale = self.scaling.detach()[candidate_indices].max(1).values
        clone_indices = candidate_indices[candidate_scale <= scale_limit]
        split_indices = candidate_indices[candidate_scale > scale_limit]
        extensions = []
        sources = []
        if len(clone_indices):
            extensions.append(
                {
                    'xyz': self._xyz.detach()[clone_indices],
                    'f_dc': self._features_dc.detach()[clone_indices],
                    'f_rest': self._features_rest.detach()[clone_indices],
                    'opacity': self._opacity.detach()[clone_indices],
                    'scaling': self._scaling.detach()[clone_indices],
                    'rotation': self._rotation.detach()[clone_indices],
                }
            )
            sources.append(clone_indices)
        if len(split_indices):
            repeats = 2
            scales = self.scaling.detach()[split_indices]
            samples = torch.randn((len(split_indices) * repeats, 3), device=self.device) * scales.repeat_interleave(repeats, dim=0)
            rotations = quaternion_to_matrix(self._rotation.detach()[split_indices]).repeat_interleave(repeats, dim=0)
            offsets = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
            split_sources = split_indices.repeat_interleave(repeats)
            extensions.append(
                {
                    'xyz': self._xyz.detach()[split_sources] + offsets,
                    'f_dc': self._features_dc.detach()[split_sources],
                    'f_rest': self._features_rest.detach()[split_sources],
                    'opacity': self._opacity.detach()[split_sources],
                    'scaling': torch.log(self.scaling.detach()[split_sources] / (0.8 * repeats)),
                    'rotation': self._rotation.detach()[split_sources],
                }
            )
            sources.append(split_sources)
        combined = {name: torch.cat([value[name] for value in extensions], dim=0) for name in self.parameter_names}
        source_indices = torch.cat(sources)
        original_count = len(self.xyz)
        added = self._append_internal(combined, source_indices, keyframe)
        prune_mask = torch.zeros(len(self.xyz), dtype=torch.bool, device=self.device)
        prune_mask[split_indices] = True
        prune_mask[:original_count] |= self.opacity.detach()[:original_count].squeeze() < min_opacity
        removed = self.prune(prune_mask)
        self.xyz_gradient_accum.zero_()
        self.gradient_denom.zero_()
        return added, removed

    def reset_opacity(self, maximum=0.01):
        """Reset excessive opacity while preserving optimizer alignment."""
        with torch.no_grad():
            reset = torch.minimum(self.opacity, torch.full_like(self.opacity, float(maximum)))
            self._opacity.copy_(inverse_sigmoid(reset))
            opacity_group = next(group for group in self.optimizer.param_groups if group['name'] == 'opacity')
            state = self.optimizer.state.get(opacity_group['params'][0])
            if state is not None:
                state['exp_avg'].zero_()
                state['exp_avg_sq'].zero_()

    def save_ply(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep exported PLYs compatible with viewers configured for SH degree
        # 3 while the online optimizer stores DC only. The padding affects
        # disk serialization, not runtime parameters or Adam state.
        rest = self._features_rest.detach().transpose(1, 2).flatten(1)
        viewer_rest_count = 3 * ((3 + 1) ** 2 - 1)
        if rest.shape[1] < viewer_rest_count:
            rest = torch.cat((rest, torch.zeros((len(rest), viewer_rest_count - rest.shape[1]), dtype=rest.dtype, device=rest.device)), dim=1)
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        names += ['f_dc_%d' % i for i in range(3)]
        names += ['f_rest_%d' % i for i in range(rest.shape[1])]
        names += ['opacity', 'scale_0', 'scale_1', 'scale_2']
        names += ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        xyz = self._xyz.detach().cpu().numpy()
        values = np.concatenate(
            (
                xyz,
                np.zeros_like(xyz),
                self._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy(),
                rest.cpu().numpy(),
                self._opacity.detach().cpu().numpy(),
                self._scaling.detach().cpu().numpy(),
                self._rotation.detach().cpu().numpy(),
            ),
            axis=1,
        )
        elements = np.empty(len(xyz), dtype=[(name, 'f4') for name in names])
        elements[:] = list(map(tuple, values))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(destination)

    def checkpoint_state(self):
        return {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'active_sh_degree': self.active_sh_degree,
            'max_radii2d': self.max_radii2d,
            'birth_keyframe': self.birth_keyframe,
            'last_seen_keyframe': self.last_seen_keyframe,
            'depth_inconsistency': self.depth_inconsistency,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'gradient_denom': self.gradient_denom,
        }

    def restore_checkpoint_state(self, state, config):
        model_state = state['model']
        for name in self.parameter_names:
            attribute = self.parameter_attributes[name]
            value = model_state[attribute].to(self.device)
            setattr(self, attribute, nn.Parameter(value.requires_grad_(True)))
        self.active_sh_degree = int(state.get('active_sh_degree', 0))
        self.max_radii2d = state['max_radii2d'].to(self.device)
        self.birth_keyframe = state['birth_keyframe'].to(self.device)
        self.last_seen_keyframe = state['last_seen_keyframe'].to(self.device)
        self.depth_inconsistency = state['depth_inconsistency'].to(self.device)
        self.xyz_gradient_accum = state.get('xyz_gradient_accum', torch.zeros(len(self.xyz))).to(self.device)
        self.gradient_denom = state.get('gradient_denom', torch.zeros(len(self.xyz))).to(self.device)
        self.setup_optimizer(config)
        if state.get('optimizer') is not None:
            self.optimizer.load_state_dict(state['optimizer'])
