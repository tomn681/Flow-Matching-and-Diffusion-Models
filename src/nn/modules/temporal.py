from __future__ import annotations

import math

import torch
import torch.nn as nn

from nn.blocks.attention import QKVAttention
from nn.blocks.common import zero_module


class TemporalAttentionND(nn.Module):
    """Self-attention across the temporal axis for video or temporal feature maps.

    Expected input shape is `(B, C, T, *spatial)`. The module flattens the spatial
    axes into the batch dimension, applies attention over `T`, and then restores the
    original layout. This keeps convolutions free to operate over `(T, *spatial)`
    while temporal attention is isolated to sequence mixing.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        *,
        dropout: float = 0.0,
        zero_init_proj_out: bool = False,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be > 0.")
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be > 0.")
        if int(channels) % int(num_heads) != 0:
            raise ValueError("channels must be divisible by num_heads.")

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads

        self.norm = nn.LayerNorm(self.channels)
        self.q_proj = nn.Linear(self.channels, self.channels)
        self.k_proj = nn.Linear(self.channels, self.channels)
        self.v_proj = nn.Linear(self.channels, self.channels)
        proj_out = nn.Linear(self.channels, self.channels)
        self.proj_out = zero_module(proj_out) if zero_init_proj_out else proj_out
        self.attention = QKVAttention(efficient_attn=True, dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 4:
            raise ValueError(
                "TemporalAttentionND expects input shaped as (B, C, T, *spatial). "
                f"Got rank {x.ndim} with shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TemporalAttentionND expected channels={self.channels}, got {x.shape[1]}."
            )

        batch, channels, frames = x.shape[:3]
        spatial_shape = x.shape[3:]
        spatial_tokens = math.prod(spatial_shape)

        residual = x
        tokens = x.permute(0, *range(3, x.ndim), 2, 1).reshape(batch * spatial_tokens, frames, channels)
        tokens = self.norm(tokens)

        q = self.q_proj(tokens).reshape(batch * spatial_tokens, frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).reshape(batch * spatial_tokens, frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).reshape(batch * spatial_tokens, frames, self.num_heads, self.head_dim).transpose(1, 2)

        out = self.attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch * spatial_tokens, frames, channels)
        out = self.proj_out(out)
        out = out.reshape(batch, *spatial_shape, frames, channels).permute(0, x.ndim - 1, x.ndim - 2, *range(1, x.ndim - 2))
        out = out.contiguous()
        return residual + out


__all__ = ["TemporalAttentionND"]
