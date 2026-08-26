# Copyright (C) 2023, Inria, GRAPHDECO research group.
# SSIM portions derive from pytorch-ssim, Copyright Evan Su, 2017 (MIT).
# See ../LICENSE.md.
"""Silhouette-masked RGB and inverse-depth supervision."""

import math

import torch
import torch.nn.functional as functional


def _ssim_window(channels, device, dtype, size=11, sigma=1.5):
    values = torch.tensor([math.exp(-((index - size // 2) ** 2) / (2 * sigma**2)) for index in range(size)], device=device, dtype=dtype)
    values /= values.sum()
    window = values[:, None] @ values[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(first, second, size=11):
    window = _ssim_window(first.shape[-3], first.device, first.dtype, size)
    first = first.unsqueeze(0) if first.ndim == 3 else first
    second = second.unsqueeze(0) if second.ndim == 3 else second
    mu_first = functional.conv2d(first, window, padding=size // 2, groups=first.shape[1])
    mu_second = functional.conv2d(second, window, padding=size // 2, groups=second.shape[1])
    first_var = functional.conv2d(first * first, window, padding=size // 2, groups=first.shape[1]) - mu_first.square()
    second_var = functional.conv2d(second * second, window, padding=size // 2, groups=second.shape[1]) - mu_second.square()
    covariance = functional.conv2d(first * second, window, padding=size // 2, groups=first.shape[1]) - mu_first * mu_second
    score = (
        (2 * mu_first * mu_second + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((mu_first.square() + mu_second.square() + 0.01**2) * (first_var + second_var + 0.03**2))
    )
    return score.mean()


def mapping_loss(rendered_rgb, rendered_inverse_depth, silhouette, camera, silhouette_threshold=0.8, lambda_dssim=0.2, depth_weight=0.02):
    """Return total loss and detached diagnostic components."""
    detached_silhouette = silhouette.detach()
    coverage = (detached_silhouette > silhouette_threshold).float()
    count = coverage.sum().clamp_min(1.0)
    rgb_error = torch.abs(rendered_rgb - camera.original_image)
    l1 = (rgb_error * coverage).sum() / (3.0 * count)
    masked_rgb = rendered_rgb * coverage + camera.original_image.detach() * (1.0 - coverage)
    rgb_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(masked_rgb, camera.original_image))
    # The rasterizer returns alpha-accumulated inverse depth.  Comparing that
    # value directly with an observed surface biases partially covered pixels
    # toward zero and can create false foreground points.  Normalize by the
    # detached accumulated alpha; coverage keeps the unstable low-alpha region
    # out of the loss entirely.
    depth_alpha = detached_silhouette
    while depth_alpha.ndim < rendered_inverse_depth.ndim:
        depth_alpha = depth_alpha.unsqueeze(0)
    normalized_inverse_depth = rendered_inverse_depth / depth_alpha.clamp_min(1e-6)
    depth_mask = camera.depth_mask * coverage
    depth_count = depth_mask.sum().clamp_min(1.0)
    depth_l1 = (torch.abs(normalized_inverse_depth - camera.inverse_depth) * depth_mask).sum() / depth_count
    total = rgb_loss + depth_weight * depth_l1
    diagnostics = {
        'loss': float(total.detach()),
        'rgb_l1': float(l1.detach()),
        'depth_l1': float(depth_l1.detach()),
        'coverage': float(coverage.mean().detach()),
    }
    return total, diagnostics
