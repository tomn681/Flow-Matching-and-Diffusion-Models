from __future__ import annotations

import math

import torch
import torch.nn as nn

from nn.ops.time_embedding import timestep_embedding


class TimestepEmbedding(nn.Module):
    """
    Two-layer timestep embedding MLP used by UNet variants.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, sample_proj_bias)
        self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        self.act = _get_activation(act_fn)
        hidden_out = out_dim if out_dim is not None else out_channels
        self.linear_2 = nn.Linear(out_channels, hidden_out, sample_proj_bias)
        self.post_act = _get_activation(post_act_fn) if post_act_fn is not None else None

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            if self.cond_proj is None:
                raise ValueError("TimestepEmbedding received `condition`, but cond_proj_dim is not configured.")
            x = x + self.cond_proj(condition)
        x = self.linear_1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.linear_2(x)
        if self.post_act is not None:
            x = self.post_act(x)
        return x


class Timesteps(nn.Module):
    """
    Hugging Face-compatible positional timestep projector.
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)
        self.scale = float(scale)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return build_timestep_features(
            timesteps,
            self.num_channels,
            max_period=10000,
            flip_sin_to_cos=self.flip_sin_to_cos,
            freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class GaussianFourierProjection(nn.Module):
    """
    Hugging Face-compatible Gaussian Fourier timestep projection.
    """

    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1.0,
        set_W_to_weight: bool = True,
        log: bool = True,
        flip_sin_to_cos: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(int(embedding_size)) * float(scale), requires_grad=False)
        self.log = bool(log)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        if set_W_to_weight:
            del self.weight
            self.W = nn.Parameter(torch.randn(int(embedding_size)) * float(scale), requires_grad=False)
            self.weight = self.W
            del self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log:
            x = torch.log(x)
        x_proj = x[:, None] * self.weight[None, :] * 2 * math.pi
        if self.flip_sin_to_cos:
            return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def build_timestep_features(
    timesteps: torch.Tensor,
    channels: int,
    *,
    max_period: int = 10000,
    flip_sin_to_cos: bool = True,
    freq_shift: float = 0,
    scale: float = 1,
) -> torch.Tensor:
    return timestep_embedding(
        timesteps,
        channels,
        max_period=max_period,
        flip_sin_to_cos=flip_sin_to_cos,
        freq_shift=freq_shift,
        scale=scale,
    )


def _get_activation(name: str | None) -> nn.Module | None:
    if name is None:
        return None
    act = str(name).lower()
    if act in {"silu", "swish"}:
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")
