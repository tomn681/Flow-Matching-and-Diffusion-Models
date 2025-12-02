"""
Normalization layers used across blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNormND(nn.Module):
    """RMSNorm over channel dimension for N-D tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=dim, keepdim=True) + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return self.weight.view(*shape) * x / rms
