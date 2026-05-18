"""Regularization losses for latent spaces."""

from __future__ import annotations

import torch


def vq_regularizer(latents: torch.Tensor) -> torch.Tensor:
    """
    VQ-GAN style regularizer that nudges latents toward zero-mean / unit-variance.

    This is a lightweight surrogate for a full codebook; it penalizes both the
    channel-wise mean and variance drift to prevent arbitrarily scaled latents.
    """
    spatial_dims = tuple(range(2, latents.ndim))
    mean = latents.mean(dim=(0, *spatial_dims), keepdim=True)
    centered = latents - mean
    var = torch.mean(centered.pow(2))
    mean_penalty = torch.mean(mean.pow(2))
    return mean_penalty + var
