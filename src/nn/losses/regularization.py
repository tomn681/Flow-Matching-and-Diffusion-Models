"""Regularization losses for latent spaces."""

from __future__ import annotations

import warnings

import torch


def latent_moment_regularizer(latents: torch.Tensor) -> torch.Tensor:
    """
    Penalize latent mean drift and variance inflation/deflation.

    This is not a codebook or VQ regularizer. It is a lightweight latent-moment
    penalty for continuous latents that encourages zero-mean, unit-variance
    activations.
    """
    spatial_dims = tuple(range(2, latents.ndim))
    mean = latents.mean(dim=(0, *spatial_dims), keepdim=True)
    centered = latents - mean
    var = torch.mean(centered.pow(2))
    mean_penalty = torch.mean(mean.pow(2))
    var_penalty = (var - 1.0).pow(2)
    return mean_penalty + var_penalty


def vq_regularizer(latents: torch.Tensor) -> torch.Tensor:
    warnings.warn(
        "vq_regularizer(...) is deprecated and misnamed. Use latent_moment_regularizer(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return latent_moment_regularizer(latents)
