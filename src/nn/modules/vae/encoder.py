"""
Convolutional encoder used by Autoencoder-style VAEs.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.blocks.residual import ResBlockND
from nn.ops.convolution import ConvND
from nn.ops.upsampling import DownsampleND
from nn.ops.normalization import make_group_norm
from .attention_factory import build_vae_attention_layer
from .stages import EncoderStage


class Encoder(nn.Module):
    """
    Hierarchical encoder with residual blocks and optional spatial attention.
    Mirrors the Stable Diffusion VAE encoder layout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        down_channels: Optional[Tuple[int, ...]] = None,
        num_res_blocks: int | Mapping[str, int] = 2,
        attn_resolutions: Tuple[int, ...] = (),
        resolution: int = 256,
        z_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: Optional[int] = None,
        attn_dim_head: Optional[int] = None,
        double_z: bool = True,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        norm_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        zero_init_last_conv: bool = True,
        attention_impl: str = "spatial",
        zero_init_attn_out: bool = True,
        use_asymmetric_padding_downsample: bool = False,
        block_factory=None,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.double_z = double_z
        self.z_channels = z_channels
        self.spatial_dims = spatial_dims
        self.emb_channels = emb_channels
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.attention_impl = str(attention_impl).lower()
        self.norm_eps = float(norm_eps)
        self.zero_init_attn_out = bool(zero_init_attn_out)
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        if emb_channels is None and use_scale_shift_norm:
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")

        channels = tuple(down_channels) if down_channels is not None else tuple(base_ch * m for m in ch_mult)
        resolved_num_res_blocks = self._resolve_num_res_blocks(num_res_blocks)

        self.conv_in = ConvND(spatial_dims, in_channels, base_ch, 3, padding=1)

        curr_res = resolution
        in_ch = base_ch
        downs: List[EncoderStage] = []
        for idx, out_ch in enumerate(channels):
            blocks = []
            attns = []
            for _ in range(resolved_num_res_blocks):
                factory = block_factory or ResBlockND
                blocks.append(
                    factory(
                        channels=in_ch,
                        emb_channels=emb_channels,
                        dropout=dropout,
                        out_channels=out_ch,
                        use_conv=False,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        spatial_dims=spatial_dims,
                        norm_eps=self.norm_eps,
                        zero_init_last_conv=zero_init_last_conv,
                    )
                )
                in_ch = out_ch
                if use_attention and (curr_res in attn_resolutions):
                    attns.append(self._build_attention_layer(in_ch))
                else:
                    attns.append(nn.Identity())
            downsample = None
            if idx != len(channels) - 1:
                downsample = DownsampleND(
                    spatial_dims,
                    in_ch,
                    use_conv=True,
                    use_asymmetric_padding=use_asymmetric_padding_downsample,
                )
                curr_res //= 2
            downs.append(EncoderStage(blocks=blocks, attns=attns, down=downsample))
        self.downs = nn.ModuleList(downs)

        self.mid_block1 = ResBlockND(
            channels=in_ch,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=in_ch,
            use_conv=False,
            use_scale_shift_norm=self.use_scale_shift_norm,
            spatial_dims=spatial_dims,
            norm_eps=self.norm_eps,
            zero_init_last_conv=zero_init_last_conv,
        )
        self.mid_attn = self._build_attention_layer(in_ch) if use_attention else nn.Identity()
        self.mid_block2 = ResBlockND(
            channels=in_ch,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=in_ch,
            use_conv=False,
            use_scale_shift_norm=self.use_scale_shift_norm,
            spatial_dims=spatial_dims,
            norm_eps=self.norm_eps,
            zero_init_last_conv=zero_init_last_conv,
        )

        groups = norm_groups if norm_groups is not None else 32
        self.norm_out = make_group_norm(in_ch, groups=groups, eps=self.norm_eps)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = ConvND(spatial_dims, in_ch, out_ch, 3, padding=1)

    @staticmethod
    def _resolve_num_res_blocks(raw: int | Mapping[str, int]) -> int:
        if isinstance(raw, Mapping):
            value = raw.get("encoder", raw.get("shared", 2))
            resolved = int(value)
        else:
            resolved = int(raw)
        if resolved < 0:
            raise ValueError("Encoder num_res_blocks must be >= 0.")
        return resolved

    def _build_attention_layer(self, channels: int) -> nn.Module:
        return build_vae_attention_layer(
            channels=channels,
            attention_impl=self.attention_impl,
            spatial_dims=self.spatial_dims,
            norm_eps=self.norm_eps,
            zero_init_attn_out=self.zero_init_attn_out,
            attn_heads=self.attn_heads,
            attn_dim_head=self.attn_dim_head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: Optional[torch.Tensor]
        if self.emb_channels is None:
            emb = None
        else:
            emb = torch.zeros(x.size(0), self.emb_channels, dtype=x.dtype, device=x.device)

        h = self.conv_in(x)
        curr = h
        for stage in self.downs:
            for block, attn in zip(stage.blocks, stage.attns):
                curr = block(curr, emb)
                curr = attn(curr)
            if stage.down is not None:
                curr = stage.down(curr)

        h = self.mid_block1(curr, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
