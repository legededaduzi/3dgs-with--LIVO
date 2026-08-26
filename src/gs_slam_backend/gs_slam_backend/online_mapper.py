"""Fixed-pose online Gaussian mapping inspired by SplaTAM."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .camera import GaussianCamera
from .frame_packet import FramePacket
from .geometry import backproject, projected_overlap, reliable_depth_mask, should_add_keyframe
from .losses import mapping_loss
from .model import GaussianMap
from .preview import PreviewVisualizer
from .renderer import render, require_rasterizer


class OnlineMapper:
    """Incrementally grow, optimize, and clean one Gaussian map."""

    def __init__(
        self,
        session_directory,
        output_directory,
        config,
        checkpoint=None,
        save_checkpoints=True,
        unique_final_ply=False,
        refine_on_close=False,
        initial_point_cloud=None,
        refinement_packets=None,
        preview=False,
        preview_depth_min=0.2,
        preview_depth_max=5.0,
    ):
        require_rasterizer()
        self.session_directory = Path(session_directory).expanduser().resolve()
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.save_checkpoints = save_checkpoints
        self.refine_on_close = refine_on_close
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.final_ply_path = self._final_ply_path(unique_final_ply)
        self.device = torch.device(config.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available to the Gaussian backend')
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        self.model = GaussianMap(config.sh_degree, config.device, config.initial_opacity)
        self.background = torch.zeros(3, device=self.device)
        self.keyframes = []
        self.keyframe_index = -1
        self.last_processed = -1
        self.last_diagnostics = {}
        self.stopped = False
        self.scene_radius = 1.0
        self.total_added = 0
        self.total_pruned = 0
        self.total_densified = 0
        self.training_step = 0
        self.pending_prune_since = None
        self.pending_prune_count = 0
        self.occupied_voxels = set()
        self.last_status = {}
        self.initial_point_cloud = initial_point_cloud
        self.refinement_packets = refinement_packets
        self.preview = PreviewVisualizer(preview_depth_min, preview_depth_max) if preview else None
        if checkpoint is not None:
            self.restore(checkpoint)
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _final_ply_path(self, unique):
        """Choose a stable, non-overwriting final map path for this run."""
        if not unique:
            return self.output_directory / 'point_cloud.ply'
        candidate = self.output_directory / ('point_cloud_%s.ply' % self.run_id)
        suffix = 1
        while candidate.exists():
            candidate = self.output_directory / ('point_cloud_%s_%02d.ply' % (self.run_id, suffix))
            suffix += 1
        return candidate

    def _stop(self, *_):
        self.stopped = True

    @staticmethod
    def _print_phase_timing(phase, elapsed_ms, **context):
        message = {'phase': phase, 'elapsed_ms': round(elapsed_ms, 2), **context}
        print(json.dumps(message, sort_keys=True), flush=True)

    def _load_observation(self, packet, load_camera=True):
        rgb, inverse_depth, metric_depth, valid = packet.load_images(self.session_directory)
        camera = None
        if load_camera:
            camera = GaussianCamera(packet, rgb, inverse_depth, valid, self.config.device)
        return dict(packet=packet, camera=camera, rgb=rgb, inverse_depth=inverse_depth, metric_depth=metric_depth, valid=valid)

    def _camera_for(self, observation):
        if observation['camera'] is None:
            observation['camera'] = GaussianCamera(
                observation['packet'], observation['rgb'], observation['inverse_depth'], observation['valid'], self.config.device
            )
        return observation['camera']

    def _render(self, camera, return_silhouette=True, active_indices=None):
        return render(
            camera,
            self.model,
            self.background,
            return_silhouette=return_silhouette,
            frustum_culling=self.config.enable_frustum_culling,
            frustum_margin=self.config.frustum_margin,
            frustum_near=self.config.frustum_near,
            frustum_far=self.config.frustum_far,
            active_indices=active_indices,
        )

    def _constrain_model(self):
        self.model.constrain_parameters(
            minimum_opacity=self.config.minimum_opacity,
            maximum_opacity=self.config.maximum_opacity,
            minimum_scale=self.config.minimum_gaussian_scale,
            maximum_scale=self.config.maximum_gaussian_scale,
            maximum_anisotropy=self.config.maximum_gaussian_anisotropy,
        )

    def _reliable_depth(self, observation):
        return reliable_depth_mask(
            observation['metric_depth'], observation['valid'], self.config.depth_mask_erosion_pixels, self.config.depth_edge_relative_threshold
        )

    def _initialize(self, observation):
        bootstrap = self.initial_point_cloud is not None
        if self.initial_point_cloud is None:
            reliable = self._reliable_depth(observation)
            points, colors, scales = backproject(
                observation['rgb'],
                observation['metric_depth'],
                reliable,
                observation['packet'].intrinsics,
                observation['packet'].T_world_camera,
                self.config.pixel_stride,
                self.config.voxel_size,
                self.config.max_new_points,
            )
        else:
            points, colors, scales = self.initial_point_cloud
            self.initial_point_cloud = None
        opacity = self.config.bootstrap_initial_opacity if bootstrap else None
        self.model.initialize(points, colors, scales, keyframe=0, opacity=opacity)
        self._constrain_model()
        self._register_voxels(points)
        self.model.setup_optimizer(self.config)
        valid_depth = observation['metric_depth'][observation['valid']]
        self.scene_radius = float(np.percentile(valid_depth, 90)) if len(valid_depth) else 1.0
        self.keyframe_index = 0
        self.keyframes.append(observation)
        self.total_added += len(points)
        return len(points)

    def _voxel_keys(self, points):
        if self.config.voxel_size <= 0 or not len(points):
            return []
        cells = np.floor(np.asarray(points) / self.config.voxel_size).astype(np.int64)
        return [tuple(cell) for cell in cells]

    def _register_voxels(self, points):
        self.occupied_voxels.update(self._voxel_keys(points))

    def _rebuild_occupied_voxels(self):
        """Synchronize the insertion index with the current Gaussian map."""
        self.occupied_voxels.clear()
        if len(self.model.xyz):
            self._register_voxels(self.model.xyz.detach().cpu().numpy())

    def _remove_occupied_candidates(self, points, colors, scales):
        """Reject candidates in voxels already occupied by the global map."""
        keys = self._voxel_keys(points)
        if not keys:
            return points, colors, scales
        keep = []
        accepted = set()
        for index, key in enumerate(keys):
            if key in self.occupied_voxels or key in accepted:
                continue
            keep.append(index)
            accepted.add(key)
        if not keep:
            return points[:0], colors[:0], scales[:0]
        keep = np.asarray(keep, dtype=np.int64)
        return points[keep], colors[keep], scales[keep]

    def _new_gaussians(self, observation, rendering):
        # Optimization can move Gaussians and pruning can remove them.  A
        # persistent append-only voxel set would otherwise reserve their old
        # cells forever and prevent genuine coverage holes from being filled.
        self._rebuild_occupied_voxels()
        silhouette = rendering['silhouette'].detach().cpu().numpy()
        accumulated_inverse = rendering['inverse_depth'].detach().squeeze().cpu().numpy()
        rendered_inverse = np.zeros_like(accumulated_inverse)
        np.divide(accumulated_inverse, silhouette, out=rendered_inverse, where=silhouette > 1e-6)
        observed_inverse = observation['inverse_depth']
        valid = self._reliable_depth(observation)
        missing = silhouette < self.config.coverage_threshold
        # Depth ordering is meaningful only after alpha normalization and in
        # pixels whose rendered surface is already well covered.  Low coverage
        # is handled exclusively by the missing-surface branch.
        confident_surface = silhouette > self.config.loss_silhouette_threshold
        closer = confident_surface & (observed_inverse > rendered_inverse)
        closer &= (observed_inverse - rendered_inverse) > (self.config.depth_relative_threshold * observed_inverse)
        selection = valid & (missing | closer)
        points, colors, scales = backproject(
            observation['rgb'],
            observation['metric_depth'],
            selection,
            observation['packet'].intrinsics,
            observation['packet'].T_world_camera,
            self.config.pixel_stride,
            self.config.voxel_size,
            self.config.max_new_points,
        )
        points, colors, scales = self._remove_occupied_candidates(points, colors, scales)
        count = self.model.append(points, colors, scales, max(self.keyframe_index, 0))
        self._constrain_model()
        self._register_voxels(points)
        self.total_added += count
        return count

    def _keyframe_overlap(self, observation, target):
        return projected_overlap(
            observation['metric_depth'],
            observation['valid'],
            observation['packet'].intrinsics,
            observation['packet'].T_world_camera,
            target['packet'].T_world_camera,
        )

    def _maybe_add_keyframe(self, observation):
        last = self.keyframes[-1] if self.keyframes else None
        overlap = self._keyframe_overlap(observation, last) if last else 0.0
        valid_depth = observation['metric_depth'][observation['valid']]
        median = float(np.median(valid_depth)) if len(valid_depth) else 1.0
        if not should_add_keyframe(
            last['packet'] if last else None,
            observation['packet'],
            overlap,
            median,
            self.config.keyframe_max_gap,
            self.config.keyframe_overlap_threshold,
            self.config.keyframe_translation_depth_ratio,
            self.config.keyframe_rotation_threshold_deg,
            self.config.keyframe_min_gap,
        ):
            return False
        self.keyframe_index += 1
        self.keyframes.append(observation)
        return True

    def _mapping_window(self, current):
        if not self.keyframes:
            return [current]
        latest = self.keyframes[-1]
        candidates = self.keyframes[:-1]
        ranked = sorted(candidates, key=lambda frame: self._keyframe_overlap(current, frame), reverse=True)
        overlap_frames = ranked[: self.config.overlap_keyframes]
        overlap_ids = {frame['packet'].sequence_id for frame in overlap_frames}
        remaining = [frame for frame in candidates if frame['packet'].sequence_id not in overlap_ids]
        history = random.sample(remaining, min(self.config.history_keyframes, len(remaining)))
        window = [current, latest, *overlap_frames, *history]
        unique = {}
        for frame in window:
            unique[frame['packet'].sequence_id] = frame
        return list(unique.values())

    def _optimizer_step(self, rendering):
        """Apply one update plus quality-training maintenance."""
        visible = rendering['visibility']
        global_visible = rendering['active_indices'][visible]
        gradient = rendering['viewspace_points'].grad
        if gradient is not None:
            self.model.add_densification_stats(global_visible, gradient[visible].detach())
        self.model.last_seen_keyframe[global_visible] = self.keyframe_index
        self.model.max_radii2d[global_visible] = torch.maximum(self.model.max_radii2d[global_visible], rendering['radii'][visible])
        self.model.update_position_learning_rate(self.training_step + 1, self.config.position_lr_final, self.config.position_lr_max_steps)
        if self.config.optimizer_type == 'sparse_adam':
            global_visibility = torch.zeros(len(self.model.xyz), dtype=torch.bool, device=self.device)
            global_visibility[global_visible] = True
            self.model.optimizer.step(global_visibility, len(self.model.xyz))
        else:
            self.model.optimizer.step()
        self._constrain_model()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.training_step += 1

        interval = self.config.sh_degree_interval
        if interval > 0 and self.training_step % interval == 0:
            self.model.active_sh_degree = min(self.model.active_sh_degree + 1, self.model.max_sh_degree)

        mutated = False
        if (
            self.config.enable_densification
            and self.training_step > self.config.densify_from_step
            and self.training_step < self.config.densify_until_step
            and self.config.densification_interval > 0
            and self.training_step % self.config.densification_interval == 0
        ):
            added, removed = self.model.densify_and_prune(
                self.config.densify_grad_threshold,
                self.config.percent_dense,
                self.scene_radius,
                self.config.densify_min_opacity,
                self.config.densify_max_points,
                keyframe=-1,
            )
            self.total_added += added
            self.total_densified += added
            self.total_pruned += removed
            mutated = bool(added or removed)

        reset_interval = self.config.opacity_reset_interval
        if (
            self.config.enable_densification
            and reset_interval > 0
            and self.training_step < self.config.densify_until_step
            and self.training_step % reset_interval == 0
        ):
            self.model.reset_opacity(self.config.opacity_reset_value)
        return mutated

    def _depth_weight(self):
        final = self.config.depth_weight_final
        if final is None or self.config.depth_weight_max_steps <= 0:
            return self.config.depth_weight
        progress = min(self.training_step / self.config.depth_weight_max_steps, 1.0)
        return float(np.exp(np.log(self.config.depth_weight) * (1.0 - progress) + np.log(final) * progress))

    def _optimize(self, observation):
        window_started = time.monotonic()
        window = self._mapping_window(observation)
        window_select_ms = (time.monotonic() - window_started) * 1000
        active_cache = {}
        started = time.monotonic()
        diagnostics = {}
        completed = 0
        preview_render = None
        for iteration in range(self.config.mapping_iterations):
            elapsed_ms = (time.monotonic() - started) * 1000
            if self.config.mapping_time_budget_ms > 0 and elapsed_ms >= self.config.mapping_time_budget_ms:
                break
            frame = window[iteration % len(window)]
            camera = self._camera_for(frame)
            sequence_id = frame['packet'].sequence_id
            rendering = self._render(camera, active_indices=active_cache.get(sequence_id))
            active_cache[sequence_id] = rendering['active_indices']
            loss, diagnostics = mapping_loss(
                rendering['render'],
                rendering['inverse_depth'],
                rendering['silhouette'],
                camera,
                self.config.loss_silhouette_threshold,
                self.config.lambda_dssim,
                self._depth_weight(),
            )
            loss.backward()
            if self.preview is not None and frame['packet'].sequence_id == observation['packet'].sequence_id:
                # Keep only a detached reference to the newest current-frame
                # training render. This avoids an extra preview rasterization.
                preview_render = rendering['render'].detach()
            render_model_count = len(self.model.xyz)
            if self._optimizer_step(rendering):
                active_cache.clear()
            diagnostics['active_gaussians'] = len(rendering['active_indices'])
            diagnostics['active_fraction'] = diagnostics['active_gaussians'] / max(render_model_count, 1)
            diagnostics['training_step'] = self.training_step
            diagnostics['active_sh_degree'] = self.model.active_sh_degree
            diagnostics['total_densified'] = self.total_densified
            completed += 1
        diagnostics['iterations'] = completed
        diagnostics['window_select_ms'] = window_select_ms
        diagnostics['mapping_ms'] = (time.monotonic() - started) * 1000
        return diagnostics, preview_render

    @staticmethod
    def _preview_bgr(rendered_rgb):
        """Quantize on the GPU so the once-per-frame transfer stays small."""
        return rendered_rgb[[2, 1, 0]].permute(1, 2, 0).mul(255).to(torch.uint8).contiguous().cpu().numpy()

    def _depth_consistency_masks(self, observation):
        """Classify map centers as free-space conflicts or depth-supported."""
        free_mask = torch.zeros(len(self.model.xyz), dtype=torch.bool, device=self.device)
        consistent_mask = torch.zeros_like(free_mask)

        packet = observation['packet']
        transform = torch.as_tensor(np.linalg.inv(np.asarray(packet.T_world_camera)), dtype=torch.float32, device=self.device)
        xyz = self.model.xyz.detach()
        camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        positive = camera_xyz[:, 2] > 0
        intrinsics = packet.intrinsics
        u = torch.round(intrinsics['fx'] * camera_xyz[:, 0] / camera_xyz[:, 2].clamp_min(1e-6) + intrinsics['cx']).long()
        v = torch.round(intrinsics['fy'] * camera_xyz[:, 1] / camera_xyz[:, 2].clamp_min(1e-6) + intrinsics['cy']).long()
        inside = positive & (u >= 0) & (u < packet.width)
        inside &= (v >= 0) & (v < packet.height)
        indices = torch.where(inside)[0]
        if not len(indices):
            return free_mask, consistent_mask
        metric_depth = torch.from_numpy(observation['metric_depth']).to(self.device)
        reliable = torch.from_numpy(self._reliable_depth(observation)).to(self.device)
        indices = indices[reliable[v[indices], u[indices]]]
        if not len(indices):
            return free_mask, consistent_mask
        observed = metric_depth[v[indices], u[indices]]
        tolerance = self.config.depth_relative_threshold * observed
        projected_depth = camera_xyz[indices, 2]
        free_space = projected_depth < observed - tolerance
        consistent = torch.abs(projected_depth - observed) <= tolerance
        free_mask[indices[free_space]] = True
        consistent_mask[indices[consistent]] = True
        return free_mask, consistent_mask

    def _update_depth_consistency(self, observation):
        """Accumulate occlusion-safe online free-space evidence.

        A Gaussian behind the measured surface is merely occluded and provides
        no pruning evidence.  Agreement clears prior evidence, while a center
        significantly in front of reliable depth increments it.
        """
        free_mask, consistent_mask = self._depth_consistency_masks(observation)
        self.model.depth_inconsistency[consistent_mask] = 0
        self.model.depth_inconsistency[free_mask] += 1

    def _final_free_space_cleanup(self, frames):
        """Remove ghosts contradicted by many archived frames before refining."""
        free_count = torch.zeros(len(self.model.xyz), dtype=torch.int32, device=self.device)
        support_count = torch.zeros_like(free_count)
        for frame in frames:
            free_mask, consistent_mask = self._depth_consistency_masks(frame)
            free_count += free_mask
            support_count += consistent_mask
        threshold = self.config.ghost_inconsistency_limit
        selected = free_count >= threshold
        # A real surface may have a few conflicts from pose/depth noise.  Require
        # free-space contradictions to clearly dominate supporting observations.
        selected &= free_count > 2 * support_count
        removed = self.model.prune(selected)
        if removed:
            self._rebuild_occupied_voxels()
        self.total_pruned += removed
        return removed

    def _prune(self, force=False):
        """Accumulate deletion candidates and compact them in bounded batches."""
        age = self.keyframe_index - self.model.birth_keyframe
        mature = age >= self.config.newborn_grace_keyframes
        ghosts = self.model.depth_inconsistency >= self.config.ghost_inconsistency_limit
        stale = (self.keyframe_index - self.model.last_seen_keyframe) >= self.config.stale_keyframes
        low_opacity = self.model.opacity.detach().squeeze() < self.config.opacity_prune_threshold
        huge = self.model.scaling.detach().max(dim=1).values > (self.config.max_scale_ratio * self.scene_radius)
        legacy = self.config.enable_pruning
        selected = torch.zeros_like(mature)
        if legacy or self.config.enable_ghost_pruning:
            selected |= ghosts
        if legacy or self.config.enable_stale_pruning:
            selected |= stale
        if legacy or self.config.enable_opacity_pruning:
            selected |= low_opacity
        if legacy or self.config.enable_scale_pruning:
            selected |= huge
        selected &= mature
        candidate_count = int(selected.sum())
        self.pending_prune_count = candidate_count
        if candidate_count == 0:
            self.pending_prune_since = None
            return 0
        if self.pending_prune_since is None:
            self.pending_prune_since = self.keyframe_index
        waited = self.keyframe_index - self.pending_prune_since
        batch_ready = candidate_count >= self.config.prune_batch_min_points
        delay_expired = waited >= self.config.prune_batch_max_keyframes
        if not force and not batch_ready and not delay_expired:
            return 0

        removed = self.model.prune(selected)
        if removed:
            self._rebuild_occupied_voxels()
        self.total_pruned += removed
        self.pending_prune_since = None
        self.pending_prune_count = 0
        return removed

    def _pruning_enabled(self):
        return bool(
            self.config.enable_pruning
            or self.config.enable_ghost_pruning
            or self.config.enable_opacity_pruning
            or self.config.enable_stale_pruning
            or self.config.enable_scale_pruning
        )

    def process(self, packet, optimize=True, queue_length=0):
        if packet.sequence_id <= self.last_processed:
            return None
        started = time.monotonic()
        timings = {}

        phase_started = time.monotonic()
        observation = self._load_observation(packet)
        timings['load'] = (time.monotonic() - phase_started) * 1000

        phase_started = time.monotonic()
        if len(self.model.xyz) == 0:
            keyframe_added = True
        else:
            # Backlogged frames remain archived on disk, but they must not add
            # unoptimized Gaussians or enter the persistent keyframe library.
            keyframe_added = self._maybe_add_keyframe(observation) if optimize else False
        timings['keyframe_decision'] = (time.monotonic() - phase_started) * 1000

        phase_started = time.monotonic()
        if len(self.model.xyz) == 0:
            added = self._initialize(observation)
        else:
            if keyframe_added:
                rendering = self._render(observation['camera'])
                added = self._new_gaussians(observation, rendering)
            else:
                added = 0
        timings['gaussian_growth'] = (time.monotonic() - phase_started) * 1000

        if optimize:
            diagnostics, preview_render = self._optimize(observation)
        else:
            diagnostics = {'iterations': 0, 'window_select_ms': 0.0, 'mapping_ms': 0.0}
            preview_render = None
        timings['window_select'] = diagnostics['window_select_ms']
        timings['optimization'] = diagnostics['mapping_ms']

        pruned = 0
        if keyframe_added and optimize and self._pruning_enabled():
            phase_started = time.monotonic()
            self._update_depth_consistency(observation)
            timings['depth_consistency'] = (time.monotonic() - phase_started) * 1000
            phase_started = time.monotonic()
            pruned = self._prune()
            timings['prune'] = (time.monotonic() - phase_started) * 1000
        else:
            timings['depth_consistency'] = 0.0
            timings['prune'] = 0.0

        phase_started = time.monotonic()
        if self.preview is not None and preview_render is not None:
            keep_running = self.preview.show(observation['rgb'], observation['metric_depth'], observation['valid'], self._preview_bgr(preview_render))
            if not keep_running:
                self.stopped = True
        timings['preview'] = (time.monotonic() - phase_started) * 1000

        self.last_processed = packet.sequence_id
        self.last_diagnostics = diagnostics
        checkpoint_started = time.monotonic()
        if self.save_checkpoints and keyframe_added and len(self.keyframes) % self.config.checkpoint_keyframes == 0:
            self.save_checkpoint()
        timings['checkpoint'] = (time.monotonic() - checkpoint_started) * 1000
        # Historical RGB-D stays on CPU. Window cameras are materialized only
        # for the mapping step so long sessions do not retain every image on GPU.
        for frame in self.keyframes:
            frame['camera'] = None
        timings['total'] = (time.monotonic() - started) * 1000

        result = {
            'last_received': packet.sequence_id,
            'last_processed': self.last_processed,
            'queue_length': queue_length,
            'gaussians': len(self.model.xyz),
            'keyframes': len(self.keyframes),
            'keyframe_added': keyframe_added,
            'keyframe_ratio': (len(self.keyframes) / max(packet.sequence_id + 1, 1)),
            'added': added,
            'pruned': pruned,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            'pending_prune': self.pending_prune_count,
            'total_densified': self.total_densified,
            'training_step': self.training_step,
            'active_sh_degree': self.model.active_sh_degree,
            'training_profile': self.config.training_profile,
            'pruning_enabled': self._pruning_enabled(),
            'pruning_modes': {
                'ghost': self.config.enable_pruning or self.config.enable_ghost_pruning,
                'opacity': self.config.enable_pruning or self.config.enable_opacity_pruning,
                'stale': self.config.enable_pruning or self.config.enable_stale_pruning,
                'scale': self.config.enable_pruning or self.config.enable_scale_pruning,
            },
            'run_id': self.run_id,
            'final_ply': self.final_ply_path.name,
            'frame_ms': timings['total'],
            'timing_ms': timings,
            **diagnostics,
        }
        self.last_status = result
        self._write_status(result)
        return result

    def _write_status(self, status):
        destination = self.output_directory / 'status.json'
        temporary = destination.with_name(destination.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(status, stream, indent=2, sort_keys=True)
            stream.write('\n')
        temporary.replace(destination)

    def save_checkpoint(self, path=None):
        destination = Path(path) if path else self.output_directory / 'latest.pth'
        temporary = destination.with_name(destination.name + '.tmp')
        payload = {
            'model': self.model.checkpoint_state(),
            'last_processed': self.last_processed,
            'keyframe_index': self.keyframe_index,
            'keyframes': [asdict(frame['packet']) for frame in self.keyframes],
            'scene_radius': self.scene_radius,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            'total_densified': self.total_densified,
            'training_step': self.training_step,
            'pending_prune_since': self.pending_prune_since,
        }
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination

    def save_final_ply(self):
        """Atomically save the final Graphdeco-compatible map."""
        temporary = self.final_ply_path.with_name(self.final_ply_path.name + '.tmp')
        self.model.save_ply(temporary)
        temporary.replace(self.final_ply_path)
        return self.final_ply_path

    def _final_refinement(self):
        """Refine the map over keyframes or every archived input frame."""
        target = self.config.final_refinement_iterations
        if target <= 0:
            return 0
        if self.config.refine_all_frames and self.refinement_packets:
            frames = [self._load_observation(packet, load_camera=False) for packet in self.refinement_packets]
        else:
            frames = self.keyframes
        if not frames:
            return 0
        started = time.monotonic()
        ghost_pruned = 0
        if self.config.enable_ghost_pruning:
            ghost_pruned = self._final_free_space_cleanup(frames)
            self.last_status = {
                **self.last_status,
                'phase': 'pre_refinement_cleanup',
                'pre_refinement_ghost_pruned': ghost_pruned,
                'gaussians': len(self.model.xyz),
                'total_pruned': self.total_pruned,
            }
            self._write_status(self.last_status)
            self._print_phase_timing('pre_refinement_cleanup', (time.monotonic() - started) * 1000, pruned=ghost_pruned)
        completed = 0
        diagnostics = {}
        repeats_per_frame = 5
        frame_order = list(range(len(frames)))
        frame_cursor = len(frame_order)
        active_frame = None
        active_indices = None
        for iteration in range(target):
            if self.config.shuffle_refinement_frames:
                if frame_cursor >= len(frame_order):
                    random.shuffle(frame_order)
                    frame_cursor = 0
                frame = frames[frame_order[frame_cursor]]
                frame_cursor += 1
            else:
                frame_index = (iteration // repeats_per_frame) % len(frames)
                frame = frames[frame_index]
            if active_frame is not None and active_frame is not frame:
                active_frame['camera'] = None
                active_indices = None
            active_frame = frame
            camera = self._camera_for(frame)
            rendering = self._render(camera, active_indices=active_indices)
            active_indices = rendering['active_indices']
            loss, diagnostics = mapping_loss(
                rendering['render'],
                rendering['inverse_depth'],
                rendering['silhouette'],
                camera,
                self.config.loss_silhouette_threshold,
                self.config.lambda_dssim,
                self._depth_weight(),
            )
            loss.backward()
            render_model_count = len(self.model.xyz)
            if self._optimizer_step(rendering):
                active_indices = None
            diagnostics['active_gaussians'] = len(rendering['active_indices'])
            diagnostics['active_fraction'] = diagnostics['active_gaussians'] / max(render_model_count, 1)
            diagnostics['training_step'] = self.training_step
            diagnostics['active_sh_degree'] = self.model.active_sh_degree
            diagnostics['total_densified'] = self.total_densified
            completed += 1
            interval = self.config.final_refinement_status_interval
            if interval > 0 and completed % interval == 0:
                status = {
                    **self.last_status,
                    **diagnostics,
                    'phase': 'final_refinement',
                    'final_refinement_iteration': completed,
                    'final_refinement_total': target,
                    'final_refinement_frames': len(frames),
                    'pre_refinement_ghost_pruned': ghost_pruned,
                    'final_refinement_ms': (time.monotonic() - started) * 1000,
                }
                self._write_status(status)
                self._print_phase_timing(
                    'final_refinement',
                    status['final_refinement_ms'],
                    iteration=completed,
                    total=target,
                )
        if active_frame is not None:
            active_frame['camera'] = None
        self.last_status = {
            **self.last_status,
            **diagnostics,
            'phase': 'saving_final_ply',
            'final_refinement_iteration': completed,
            'final_refinement_total': target,
            'final_refinement_frames': len(frames),
            'pre_refinement_ghost_pruned': ghost_pruned,
            'final_refinement_ms': (time.monotonic() - started) * 1000,
        }
        self._write_status(self.last_status)
        self._print_phase_timing(
            'saving_final_ply',
            self.last_status['final_refinement_ms'],
            iteration=completed,
            total=target,
        )
        return completed

    def _final_cleanup(self):
        if not self.config.enable_final_cleanup:
            return 0
        low_opacity = self.model.opacity.detach().squeeze() < (self.config.final_opacity_prune_threshold)
        oversized = self.model.scaling.detach().max(dim=1).values > (self.config.final_maximum_gaussian_scale)
        removed = self.model.prune(low_opacity | oversized)
        if removed:
            self._rebuild_occupied_voxels()
        self.total_pruned += removed
        return removed

    def restore(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.restore_checkpoint_state(payload['model'], self.config)
        self._constrain_model()
        self.last_processed = int(payload['last_processed'])
        self.keyframe_index = int(payload['keyframe_index'])
        self.scene_radius = float(payload['scene_radius'])
        self.total_added = int(payload.get('total_added', len(self.model.xyz)))
        self.total_pruned = int(payload.get('total_pruned', 0))
        self.total_densified = int(payload.get('total_densified', 0))
        self.training_step = int(payload.get('training_step', 0))
        self.pending_prune_since = payload.get('pending_prune_since')
        self.pending_prune_count = 0
        self.keyframes = [self._load_observation(FramePacket(**value).validate(), load_camera=False) for value in payload['keyframes']]
        self._register_voxels(self.model.xyz.detach().cpu().numpy())

    def close(self):
        if self.preview is not None:
            self.preview.close()
        if len(self.model.xyz):
            try:
                self._prune(force=True)
                if self.refine_on_close:
                    self._final_refinement()
                final_pruned = self._final_cleanup()
                self.last_status = {
                    **self.last_status,
                    'gaussians': len(self.model.xyz),
                    'final_pruned': final_pruned,
                    'total_pruned': self.total_pruned,
                    'phase': 'saving_final_ply',
                }
                self._write_status(self.last_status)
                if self.save_checkpoints:
                    self.save_checkpoint()
            finally:
                self.save_final_ply()
                self.last_status = {**self.last_status, 'phase': 'complete', 'final_ply': self.final_ply_path.name}
                self._write_status(self.last_status)
