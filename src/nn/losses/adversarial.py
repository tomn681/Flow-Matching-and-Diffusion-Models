"""Adversarial losses and discriminators used in VAE/GAN-style training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.ops.convolution import ConvND


class PatchDiscriminator(nn.Module):
    """Small PatchGAN-style discriminator used by the VAE."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64, spatial_dims: int = 2) -> None:
        super().__init__()
        ch = base_channels
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")
        bn_map = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
        Norm = bn_map[spatial_dims]
        self.model = nn.Sequential(
            ConvND(spatial_dims, in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch, ch * 2, 4, 2, 1),
            Norm(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 2, ch * 4, 4, 2, 1),
            Norm(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 4, ch * 8, 4, 2, 1),
            Norm(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvND(spatial_dims, ch * 8, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def discriminator_hinge_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Standard hinge loss for the discriminator."""
    return torch.mean(F.relu(1.0 - real_pred)) + torch.mean(F.relu(1.0 + fake_pred))


def generator_hinge_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss that encourages fake predictions to be "real"."""
    return -torch.mean(fake_pred)
