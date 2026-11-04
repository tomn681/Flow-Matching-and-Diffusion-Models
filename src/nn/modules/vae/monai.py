"""MONAI 1.6.1 AutoencoderKL topology built from reusable VAE blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from nn.blocks.attention import QKVAttention, SpatialSelfAttention
from nn.blocks.residual import ResBlockND
from .decoder import Decoder
from .encoder import Encoder


class MonaiResBlock(ResBlockND):
    """Residual block specialization matching MONAI AutoencoderKL."""

    def __init__(self, **kwargs) -> None:
        kwargs.update(
            emb_channels=None,
            dropout=0.0,
            norm_type="gn",
            act="silu",
            zero_init_last_conv=False,
            output_scale_factor=1.0,
        )
        super().__init__(**kwargs)
        self.dropout_layer = nn.Identity()


class MonaiSelfAttention(nn.Module):
    """Separate Q/K/V projection used by MONAI's spatial attention block."""

    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool) -> None:
        super().__init__()
        if not batch_first or embed_dim % num_heads:
            raise ValueError("MONAI attention requires batch-first input and channels divisible by heads.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.attention = QKVAttention(efficient_attn=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _project(self, tensor: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return projection(tensor).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        hidden = self.attention(
            self._project(query, self.to_q),
            self._project(key, self.to_k),
            self._project(value, self.to_v),
        )
        batch, _, tokens, _ = hidden.shape
        return self.out_proj(hidden.transpose(1, 2).reshape(batch, tokens, -1)), None


class MonaiSpatialAttention(SpatialSelfAttention):
    """Spatial attention specialization matching MONAI AutoencoderKL."""

    def __init__(
        self,
        channels: int,
        *,
        spatial_dims: int,
        norm_groups: int,
        norm_eps: float,
    ) -> None:
        super().__init__(
            channels,
            spatial_dims=spatial_dims,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
            attention_factory=MonaiSelfAttention,
        )


class _MonaiTopology:
    """Shared topology overrides for the MONAI encoder and decoder."""

    def _build_attention_layer(self, channels: int) -> nn.Module:
        return MonaiSpatialAttention(
            channels,
            spatial_dims=self.spatial_dims,
            norm_groups=self.norm_groups,
            norm_eps=self.norm_eps,
        )

    def _build_mid_resblock(self, channels: int) -> nn.Module:
        return self._build_resblock(channels, channels)

    @staticmethod
    def _use_level_attention(block_index: int, num_blocks: int) -> bool:
        return block_index == num_blocks - 1


class MonaiEncoder(_MonaiTopology, Encoder):
    """Encoder with the exact MONAI AutoencoderKL block ordering."""


class MonaiDecoder(_MonaiTopology, Decoder):
    """Decoder with the exact MONAI AutoencoderKL block ordering."""

    @staticmethod
    def _stage_num_res_blocks(num_res_blocks: int) -> int:
        return num_res_blocks
