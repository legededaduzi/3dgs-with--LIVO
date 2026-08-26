"""Validated, grouped configuration for the Gaussian mapper."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class MapperConfig:
    """Flat runtime values with a grouped on-disk representation."""

    training_profile: str = 'default'
    device: str = 'cuda:0'
    enable_pruning: bool = False
    enable_ghost_pruning: bool = False
    enable_opacity_pruning: bool = False
    enable_stale_pruning: bool = False
    enable_scale_pruning: bool = False
    sh_degree: int = 0
    sh_degree_interval: int = 1000
    enable_frustum_culling: bool = True
    frustum_margin: float = 0.2
    frustum_near: float = 0.01
    frustum_far: float = 100.0
    pixel_stride: int = 2
    voxel_size: float = 0.03
    max_new_points: int = 50000
    depth_mask_erosion_pixels: int = 1
    depth_edge_relative_threshold: float = 0.08
    coverage_threshold: float = 0.5
    loss_silhouette_threshold: float = 0.8
    depth_relative_threshold: float = 0.05
    keyframe_overlap_threshold: float = 0.65
    keyframe_translation_depth_ratio: float = 0.05
    keyframe_rotation_threshold_deg: float = 10.0
    keyframe_min_gap: int = 0
    keyframe_max_gap: int = 5
    overlap_keyframes: int = 6
    history_keyframes: int = 2
    mapping_iterations: int = 60
    mapping_time_budget_ms: float = 800.0
    final_refinement_iterations: int = 0
    final_refinement_status_interval: int = 100
    refine_all_frames: bool = False
    shuffle_refinement_frames: bool = False
    checkpoint_keyframes: int = 10
    newborn_grace_keyframes: int = 3
    ghost_inconsistency_limit: int = 3
    prune_batch_min_points: int = 1000
    prune_batch_max_keyframes: int = 10
    stale_keyframes: int = 30
    opacity_prune_threshold: float = 0.005
    max_scale_ratio: float = 0.1
    initial_opacity: float = 0.6
    bootstrap_initial_opacity: float = 0.1
    minimum_opacity: float = 0.001
    maximum_opacity: float = 0.99
    minimum_gaussian_scale: float = 0.001
    maximum_gaussian_scale: float = 0.05
    maximum_gaussian_anisotropy: float = 0.0
    enable_final_cleanup: bool = True
    final_opacity_prune_threshold: float = 0.005
    final_maximum_gaussian_scale: float = 0.05
    lambda_dssim: float = 0.2
    depth_weight: float = 0.02
    depth_weight_final: float | None = None
    depth_weight_max_steps: int = 5000
    optimizer_type: str = 'adam'
    position_lr: float = 0.0001
    position_lr_final: float | None = None
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.001
    rotation_lr: float = 0.001
    enable_densification: bool = False
    densify_from_step: int = 500
    densify_until_step: int = 15000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    densify_min_opacity: float = 0.005
    percent_dense: float = 0.01
    densify_max_points: int = 20000
    opacity_reset_interval: int = 0
    opacity_reset_value: float = 0.01
    random_seed: int = 0

    _GROUPS = {
        'runtime': ('training_profile', 'device', 'random_seed'),
        'initialization': ('pixel_stride', 'voxel_size', 'max_new_points', 'initial_opacity', 'bootstrap_initial_opacity'),
        'geometry': (
            'depth_mask_erosion_pixels',
            'depth_edge_relative_threshold',
            'coverage_threshold',
            'depth_relative_threshold',
            'minimum_gaussian_scale',
            'maximum_gaussian_scale',
            'maximum_gaussian_anisotropy',
            'minimum_opacity',
            'maximum_opacity',
        ),
        'keyframes': (
            'keyframe_overlap_threshold',
            'keyframe_translation_depth_ratio',
            'keyframe_rotation_threshold_deg',
            'keyframe_min_gap',
            'keyframe_max_gap',
            'overlap_keyframes',
            'history_keyframes',
            'checkpoint_keyframes',
        ),
        'optimization': (
            'mapping_iterations',
            'mapping_time_budget_ms',
            'loss_silhouette_threshold',
            'lambda_dssim',
            'depth_weight',
            'depth_weight_final',
            'depth_weight_max_steps',
            'optimizer_type',
            'position_lr',
            'position_lr_final',
            'position_lr_max_steps',
            'feature_lr',
            'opacity_lr',
            'scaling_lr',
            'rotation_lr',
            'sh_degree',
            'sh_degree_interval',
        ),
        'refinement': ('final_refinement_iterations', 'final_refinement_status_interval', 'refine_all_frames', 'shuffle_refinement_frames'),
        'pruning': ('prune_batch_min_points', 'prune_batch_max_keyframes'),
    }
    _FEATURES = {
        'frustum_culling': ('enable_frustum_culling', ('frustum_margin', 'frustum_near', 'frustum_far')),
        'densification': (
            'enable_densification',
            (
                'densify_from_step',
                'densify_until_step',
                'densification_interval',
                'densify_grad_threshold',
                'densify_min_opacity',
                'percent_dense',
                'densify_max_points',
            ),
        ),
        'opacity_reset': (None, ('opacity_reset_interval', 'opacity_reset_value')),
        'legacy_pruning': ('enable_pruning', ()),
        'ghost_pruning': ('enable_ghost_pruning', ('newborn_grace_keyframes', 'ghost_inconsistency_limit')),
        'opacity_pruning': ('enable_opacity_pruning', ('opacity_prune_threshold',)),
        'stale_pruning': ('enable_stale_pruning', ('stale_keyframes',)),
        'scale_pruning': ('enable_scale_pruning', ('max_scale_ratio',)),
        'final_cleanup': ('enable_final_cleanup', ('final_opacity_prune_threshold', 'final_maximum_gaussian_scale')),
    }

    @classmethod
    def load(cls, path=None):
        """Load grouped JSON, while accepting the old flat format."""
        if path is None:
            return cls()
        with Path(path).expanduser().open(encoding='utf-8') as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError('Backend configuration must be a JSON object')
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in document.items() if key in known}
        consumed = set(values) | {'schema_version'}

        for group, names in cls._GROUPS.items():
            section = document.get(group)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ValueError('%s must be a JSON object' % group)
            unknown = set(section) - set(names)
            if unknown:
                raise ValueError('Unknown %s parameter(s): %s' % (group, ', '.join(sorted(unknown))))
            values.update(section)
            consumed.add(group)

        features = document.get('features', {})
        if not isinstance(features, dict):
            raise ValueError('features must be a JSON object')
        unknown_features = set(features) - set(cls._FEATURES)
        if unknown_features:
            raise ValueError('Unknown feature(s): %s' % ', '.join(sorted(unknown_features)))
        for name, section in features.items():
            if not isinstance(section, dict):
                raise ValueError('features.%s must be a JSON object' % name)
            switch, parameters = cls._FEATURES[name]
            unknown = set(section) - {'enabled', 'parameters'}
            if unknown:
                raise ValueError('Unknown features.%s field(s): %s' % (name, ', '.join(sorted(unknown))))
            if switch is not None and 'enabled' in section:
                values[switch] = section['enabled']
            parameter_values = section.get('parameters', {})
            if not isinstance(parameter_values, dict):
                raise ValueError('features.%s.parameters must be a JSON object' % name)
            unknown = set(parameter_values) - set(parameters)
            if unknown:
                raise ValueError('Unknown features.%s parameter(s): %s' % (name, ', '.join(sorted(unknown))))
            values.update(parameter_values)
            if name == 'opacity_reset' and section.get('enabled') is False:
                values['opacity_reset_interval'] = 0
        if features:
            consumed.add('features')

        unknown = set(document) - consumed
        if unknown:
            raise ValueError('Unknown backend configuration section(s): %s' % ', '.join(sorted(unknown)))
        config = cls(**values)
        return config.validate()

    def validate(self):
        """Reject common mistakes before CUDA work starts."""
        if self.pixel_stride < 1:
            raise ValueError('initialization.pixel_stride must be >= 1')
        if self.voxel_size <= 0:
            raise ValueError('initialization.voxel_size must be > 0')
        if self.mapping_iterations < 0 or self.final_refinement_iterations < 0:
            raise ValueError('iteration counts must be non-negative')
        if self.optimizer_type not in {'adam', 'sparse_adam'}:
            raise ValueError("optimization.optimizer_type must be 'adam' or 'sparse_adam'")
        if self.prune_batch_min_points < 1:
            raise ValueError('pruning.prune_batch_min_points must be >= 1')
        if self.prune_batch_max_keyframes < 1:
            raise ValueError('pruning.prune_batch_max_keyframes must be >= 1')
        if self.minimum_opacity <= 0 or self.maximum_opacity > 1:
            raise ValueError('opacity bounds must be within (0, 1]')
        if self.minimum_opacity >= self.maximum_opacity:
            raise ValueError('minimum_opacity must be below maximum_opacity')
        return self

    def to_dict(self):
        """Return the canonical grouped representation."""
        flat = asdict(self)
        document = {'schema_version': 2}
        for group, names in self._GROUPS.items():
            document[group] = {name: flat[name] for name in names}
        document['features'] = {}
        for name, (switch, parameters) in self._FEATURES.items():
            feature = {'parameters': {parameter: flat[parameter] for parameter in parameters}}
            if switch is not None:
                feature['enabled'] = flat[switch]
            elif name == 'opacity_reset':
                feature['enabled'] = flat['opacity_reset_interval'] > 0
            document['features'][name] = feature
        return document

    def write(self, path):
        destination = Path(path).expanduser()
        with destination.open('w', encoding='utf-8') as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=False)
            stream.write('\n')
