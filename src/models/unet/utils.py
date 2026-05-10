from __future__ import annotations

import torch
import torch.nn as nn

from nn.ops.time_embedding import timestep_embedding


class TimestepEmbedding(nn.Module):
    """
    Two-layer timestep embedding MLP used by UNet variants.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


def build_timestep_features(
    timesteps: torch.Tensor,
    channels: int,
    *,
    max_period: int = 10000,
    flip_sin_to_cos: bool = True,
    freq_shift: int = 0,
) -> torch.Tensor:
    return timestep_embedding(
        timesteps,
        channels,
        max_period=max_period,
        flip_sin_to_cos=flip_sin_to_cos,
        freq_shift=freq_shift,
    )
