"""
Discriminators for VAEs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nn.ops.convolution import ConvND


class MagvitDiscriminatorND(nn.Module):
    """
    MAGVIT-style discriminator that supports 1D/2D/3D inputs.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, spatial_dims: int = 2) -> None:
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        norm_map = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
        Norm = norm_map[spatial_dims]
        ch = base_channels
        self.model = nn.Sequential(
            ConvND(spatial_dims, in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch, ch * 2, 4, 2, 1),
            Norm(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 2, ch * 4, 4, 2, 1),
            Norm(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 4, ch * 8, 4, 1, 1),
            Norm(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 8, 1, 4, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MagvitDiscriminator(MagvitDiscriminatorND):
    """Backward-compatible 2D alias."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels, spatial_dims=2)
