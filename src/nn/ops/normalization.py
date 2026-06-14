"""
Normalization layers used across blocks.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


def make_group_norm(channels: int, groups: int = 32, eps: float = 1e-5) -> nn.GroupNorm:
    """
    Build GroupNorm with safe group fallback when `channels` is not divisible
    by the requested number of groups.
    """
    requested_groups = min(groups, channels)
    num_groups = requested_groups
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    if num_groups != requested_groups:
        warnings.warn(
            f"GroupNorm requested groups={requested_groups} for channels={channels}, "
            f"falling back to groups={num_groups} for divisibility.",
            stacklevel=2,
        )
    return nn.GroupNorm(num_groups, channels, eps=eps)


class RMSNormND(nn.Module):
    """RMSNorm over channel dimension for N-D tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return self.weight.view(*shape) * x / rms
