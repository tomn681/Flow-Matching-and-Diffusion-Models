"""
Convolutional decoder used by Autoencoder-style VAEs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.blocks.attention import SpatialSelfAttention
from nn.blocks.residual import ResBlockND
from nn.ops.convolution import ConvND
from nn.ops.upsampling import UpsampleND


class Decoder(nn.Module):
    """
    Hierarchical decoder with residual blocks and optional spatial attention.
    Mirrors the Stable Diffusion VAE decoder layout.
    """

    def __init__(
        self,
        out_ch: int = 3,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        down_channels: Optional[Tuple[int, ...]] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        resolution: int = 256,
        z_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: Optional[int] = None,
        attn_dim_head: Optional[int] = None,
        tanh_out: bool = False,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        block_factory=None,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.tanh_out = tanh_out
        self.spatial_dims = spatial_dims
        self.emb_channels = emb_channels
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        if emb_channels is None and use_scale_shift_norm:
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")

        channels = tuple(down_channels) if down_channels is not None else tuple(base_ch * m for m in ch_mult)

        lowest_res = resolution // (2 ** (len(channels) - 1))
        block_in = channels[-1]

        self.conv_in = ConvND(spatial_dims, z_channels, block_in, 3, padding=1)

        self.mid_block1 = ResBlockND(
            channels=block_in,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=block_in,
            use_conv=False,
            use_scale_shift_norm=self.use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )
        self.mid_attn = self._build_attention_layer(block_in) if use_attention else nn.Identity()
        self.mid_block2 = ResBlockND(
            channels=block_in,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=block_in,
            use_conv=False,
            use_scale_shift_norm=self.use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )

        ups: List[nn.Module] = []
        in_ch = block_in
        curr_res = lowest_res
        for idx, out_ch_stage in enumerate(reversed(channels)):
            blocks = []
            attns = []
            for _ in range(num_res_blocks + 1):
                factory = block_factory or ResBlockND
                blocks.append(
                    factory(
                        channels=in_ch,
                        emb_channels=emb_channels,
                        dropout=dropout,
                        out_channels=out_ch_stage,
                        use_conv=False,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        spatial_dims=spatial_dims,
                    )
                )
                in_ch = out_ch_stage
                if use_attention and (curr_res in attn_resolutions):
                    attns.append(self._build_attention_layer(in_ch))
            stage = nn.Module()
            stage.blocks = nn.ModuleList(blocks)
            stage.attns = nn.ModuleList(attns)
            if idx != len(channels) - 1:
                stage.up = UpsampleND(spatial_dims, in_ch, use_conv=True)
                curr_res *= 2
            ups.insert(0, stage)
        self.ups = nn.ModuleList(ups)

        computed_groups = max(1, torch.gcd(torch.tensor(in_ch), torch.tensor(32)).item())
        groups = norm_groups if norm_groups is not None else computed_groups
        self.norm_out = nn.GroupNorm(groups, in_ch)
        self.conv_out = ConvND(spatial_dims, in_ch, out_ch, 3, padding=1)

    def _build_attention_layer(self, channels: int) -> nn.Module:
        heads = self.attn_heads if self.attn_heads is not None else 1
        if self.attn_dim_head is not None:
            dim_head = self.attn_dim_head
        elif heads == 1:
            dim_head = channels
        else:
            dim_head = max(1, channels // heads)
        return SpatialSelfAttention(
            dim=channels,
            heads=heads,
            dim_head=dim_head,
            use_linear=False,
            use_efficient_attn=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        emb: Optional[torch.Tensor]
        if self.emb_channels is None:
            emb = None
        else:
            emb = torch.zeros(z.size(0), self.emb_channels, dtype=z.dtype, device=z.device)

        h = self.conv_in(z)
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        curr = h
        for stage in reversed(self.ups):
            for i, block in enumerate(stage.blocks):
                curr = block(curr, emb)
                if i < len(stage.attns):
                    curr = stage.attns[i](curr)
            if hasattr(stage, "up"):
                curr = stage.up(curr)

        h = F.silu(self.norm_out(curr))
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h
