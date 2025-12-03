"""
Discriminators for VAEs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MagvitDiscriminator(nn.Module):
    """
    2D discriminator inspired by MAGVIT/TATS image branch.
    Conv4x4 stride2 stacks with BatchNorm + LeakyReLU.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels
        layers = [
            nn.Conv2d(in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, 4, 1, 1),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 8, 1, 4, 1, 0),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
