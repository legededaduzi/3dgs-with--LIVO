# Copyright (C) 2023, Inria, GRAPHDECO research group.
# All rights reserved. Research/evaluation use under ../LICENSE.md.
"""Graphdeco CUDA renderer with optional silhouette output."""

import math

import torch

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    except ImportError:
        _SEPARATE_SH_AVAILABLE = False
    else:
        _SEPARATE_SH_AVAILABLE = True
except ImportError as error:  # Keep CPU-only schema tools importable.
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _SEPARATE_SH_AVAILABLE = False
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


def require_rasterizer():
    if GaussianRasterizer is None:
        raise RuntimeError('diff_gaussian_rasterization is unavailable; run the backend ' 'inside the Gaussian Conda environment') from _IMPORT_ERROR


def frustum_indices(camera, model, margin=0.2, near=0.01, far=100.0):
    """Return global indices whose centers lie in an expanded view frustum."""
    with torch.no_grad():
        xyz = model.xyz.detach()
        world_view = camera.world_view_transform
        camera_xyz = xyz @ world_view[:3, :3] + world_view[3, :3]
        z = camera_xyz[:, 2]
        horizontal = z * math.tan(camera.fov_x * 0.5) * (1.0 + margin)
        vertical = z * math.tan(camera.fov_y * 0.5) * (1.0 + margin)
        selected = (z > near) & (z < far)
        selected &= camera_xyz[:, 0].abs() <= horizontal
        selected &= camera_xyz[:, 1].abs() <= vertical
        return torch.where(selected)[0]


def _empty_render(camera, model, background):
    """Return a differentiable empty rendering for a disjoint camera."""
    anchor = model.xyz.sum() * 0.0
    image = background[:, None, None].expand(3, camera.image_height, camera.image_width) + anchor
    inverse_depth = torch.zeros((1, camera.image_height, camera.image_width), dtype=model.xyz.dtype, device=model.xyz.device) + anchor
    return {
        'render': image,
        'inverse_depth': inverse_depth,
        'silhouette': torch.zeros((camera.image_height, camera.image_width), dtype=model.xyz.dtype, device=model.xyz.device),
        'viewspace_points': torch.empty((0, 3), dtype=model.xyz.dtype, device=model.xyz.device),
        'radii': torch.empty(0, dtype=model.xyz.dtype, device=model.xyz.device),
        'visibility': torch.empty(0, dtype=torch.bool, device=model.xyz.device),
        'active_indices': torch.empty(0, dtype=torch.long, device=model.xyz.device),
    }


def render(
    camera,
    model,
    background,
    return_silhouette=True,
    frustum_culling=False,
    frustum_margin=0.2,
    frustum_near=0.01,
    frustum_far=100.0,
    active_indices=None,
):
    """Render RGB and accumulated inverse depth from one fixed camera."""
    require_rasterizer()
    if active_indices is not None:
        active = active_indices
    elif frustum_culling:
        active = frustum_indices(camera, model, frustum_margin, frustum_near, frustum_far)
    else:
        active = torch.arange(len(model.xyz), dtype=torch.long, device=model.xyz.device)
    if not len(active):
        return _empty_render(camera, model, background)
    parameters = model.render_subset(active)
    xyz = parameters['xyz']
    features = parameters['features']
    features_dc = parameters['features_dc']
    features_rest = parameters['features_rest']
    opacity = parameters['opacity']
    scaling = parameters['scaling']
    rotation = parameters['rotation']
    means2d = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)
    means2d.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.fov_x * 0.5),
        tanfovy=math.tan(camera.fov_y * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    rasterizer_arguments = dict(
        means3D=xyz, means2D=means2d, colors_precomp=None, opacities=opacity, scales=scaling, rotations=rotation, cov3D_precomp=None
    )
    if _SEPARATE_SH_AVAILABLE:
        rasterizer_arguments.update(dc=features_dc, shs=features_rest)
    else:
        rasterizer_arguments['shs'] = features
    image, radii, inverse_depth = rasterizer(**rasterizer_arguments)
    silhouette = None
    if return_silhouette:
        silhouette_settings = settings._replace(bg=torch.zeros_like(background))
        silhouette_rasterizer = GaussianRasterizer(raster_settings=silhouette_settings)
        with torch.no_grad():
            silhouette_rgb, _, _ = silhouette_rasterizer(
                means3D=xyz,
                means2D=means2d.detach(),
                shs=None,
                colors_precomp=torch.ones_like(xyz),
                opacities=opacity,
                scales=scaling,
                rotations=rotation,
                cov3D_precomp=None,
            )
            silhouette = silhouette_rgb[0].clamp(0, 1)
    return {
        'render': image.clamp(0, 1),
        'inverse_depth': inverse_depth,
        'silhouette': silhouette,
        'viewspace_points': means2d,
        'radii': radii,
        'visibility': radii > 0,
        'active_indices': active,
    }
