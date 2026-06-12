import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.unet.base import BaseUNetND
from models.unet.utils import build_timestep_features
from ..registry import MODEL_REGISTRY
from nn.blocks.residual import ResBlockND, zero_module
from nn.blocks.timestep import TimestepBlock
from nn.blocks.attention import (
    ContextBlock,
    LegacyQKVSpatialCrossAttention,
    QKVSpatialSelfAttention,
)
from nn.ops.convolution import ConvND
from nn.ops.upsampling import UpsampleND, DownsampleND
from nn.ops.pooling import PoolND, UnPoolND
from nn.ops.normalization import make_group_norm


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential that forwards timestep embeddings to children supporting TimestepBlock.

    Forward:
        x -> [torch.Tensor] (N, C, L) | (N, C, H, W) | (N, C, D, H, W)
        emb -> [torch.Tensor] (N, emb_channels)
    """
    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


@MODEL_REGISTRY.register("efficient_unet")
class EfficientUNetND(BaseUNetND):
    """
    N-Dimensional Efficient UNet with optional attention blocks.

    Attributes:
        spatial_dims     -> [int] Spatial dimensionality: 1, 2 or 3.
        in_channels      -> [int] Input channels.
        model_channels   -> [int] Base channel count.
        out_channels     -> [int] Output channels.
        num_res_blocks   -> [int] Residual blocks per resolution level.
        attention_resolutions -> [list[int] | tuple[int]] Downsample factors for self-attention blocks.
        dropout          -> [float] Dropout rate for ResBlocks.
        channel_mult     -> [tuple[int, ...]] Per-level channel multipliers.
        conv_resample    -> [bool] Use learned convs for up/down-sampling.
        dim_head         -> [int] Per-head embedding dimension in attention.
        num_heads        -> [int] Number of attention heads.
        use_linear_attn  -> [bool] If True, use linear attention variant where available.
        use_scale_shift_norm -> [bool] Use Scale-Shift conditioning in ResBlocks.
        pool_factor      -> [int] Optional input downscale (patchify) factor; 1 disables pooling.
        cross_attention_resolutions -> [list[int] | tuple[int]] Downsample factors for cross-attention blocks.
        cross_attention_dim -> [int] Context channel count for cross-attention. Defaults to 4 (VAE latents).
        cross_attention_in_middle -> [bool] Force cross-attention in the middle block.

    Notes:
        - All convolutions/upsamples/downsamps are ND envelopes (ConvND, UpsampleND, DownsampleND, PoolND, UnPoolND).
        - `QKVSpatialSelfAttention` is the canonical fused-QKV spatial attention block.
        - `LegacyQKVSpatialSelfAttention` remains available separately only for checkpoint compatibility.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        conv_resample: bool = True,
        dim_head: int = 64,
        num_heads: int = 4,
        use_linear_attn: bool = False,
        use_scale_shift_norm: bool = True,
        pool_factor: int = 1,
        cross_attention_resolutions: Optional[Sequence[int]] = None,
        cross_attention_dim: int = 4,
        cross_attention_in_middle: bool = False,
        emb_activation_before_proj: bool = False,
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        # --- config ---
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        if cross_attention_resolutions is None:
            self.cross_attention_resolutions = ()
        else:
            self.cross_attention_resolutions = tuple(cross_attention_resolutions)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.pool_factor = pool_factor
        self.cross_attention_dim = cross_attention_dim
        self.cross_attention_in_middle = cross_attention_in_middle
        self.emb_activation_before_proj = emb_activation_before_proj

        # --- time embedding ---
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- optional input pooling (patchify) ---
        if pool_factor > 1:
            self.pool = PoolND(spatial_dims, in_channels, model_channels, pool_factor)
            start_channels = model_channels
        else:
            self.pool = nn.Identity()
            start_channels = in_channels

        # --- encoder ---
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                ConvND(spatial_dims, start_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans: list[int] = [model_channels]
        ch = model_channels
        ds = 1  # running downsample factor

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlockND(
                        spatial_dims=spatial_dims,
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        emb_activation_before_proj=emb_activation_before_proj,
                    )
                ]
                ch = mult * model_channels

                if ds in self.attention_resolutions:
                    layers.append(
                        QKVSpatialSelfAttention(
                            dim=ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )
                if ds in self.cross_attention_resolutions:
                    layers.append(
                        LegacyQKVSpatialCrossAttention(
                            dim=ch,
                            context_dim=cross_attention_dim,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # add downsample between levels (except last)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        DownsampleND(spatial_dims, ch, use_conv=conv_resample)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # --- bottleneck / middle ---
        middle_layers: list[nn.Module] = [
            ResBlockND(
                spatial_dims=spatial_dims,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                emb_activation_before_proj=emb_activation_before_proj,
            ),
            QKVSpatialSelfAttention(
                ch,
                heads=num_heads,
                dim_head=dim_head,
                use_linear=False,
                use_efficient_attn=True,
            ),
        ]
        if self.cross_attention_in_middle or ds in self.cross_attention_resolutions:
            middle_layers.append(
                LegacyQKVSpatialCrossAttention(
                    dim=ch,
                    context_dim=cross_attention_dim,
                    heads=num_heads,
                    dim_head=dim_head,
                    use_linear=False,
                    use_efficient_attn=True,
                )
            )
        middle_layers.append(
            ResBlockND(
                spatial_dims=spatial_dims,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                emb_activation_before_proj=emb_activation_before_proj,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        # --- decoder ---
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # concat with skip from encoder
                layers: list[nn.Module] = [
                    ResBlockND(
                        spatial_dims=spatial_dims,
                        channels=ch + input_block_chans.pop(),
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        emb_activation_before_proj=emb_activation_before_proj,
                    )
                ]
                ch = model_channels * mult

                if ds in self.attention_resolutions:
                    layers.append(
                        QKVSpatialSelfAttention(
                            dim=ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )
                if ds in self.cross_attention_resolutions:
                    layers.append(
                        LegacyQKVSpatialCrossAttention(
                            dim=ch,
                            context_dim=cross_attention_dim,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )

                # upsample at the end of each level (except final)
                if level and i == num_res_blocks:
                    layers.append(UpsampleND(spatial_dims, ch, use_conv=conv_resample))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # --- output projection (and optional unpool) ---
        if pool_factor > 1:
            self.out = nn.Sequential(
                make_group_norm(ch, groups=32),
                nn.SiLU(),
                ConvND(spatial_dims, model_channels, model_channels, 3, padding=1),
            )
            self.unpool = UnPoolND(spatial_dims, model_channels, out_channels, pool_factor)
        else:
            self.out = nn.Sequential(
                make_group_norm(ch, groups=32),
                nn.SiLU(),
                zero_module(ConvND(spatial_dims, model_channels, out_channels, 3, padding=1)),
            )
            self.unpool = nn.Identity()

    def _prepare_input(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        context_ca: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if context_ca is not None and not (self.cross_attention_resolutions or self.cross_attention_in_middle):
            raise ValueError("context_ca provided but cross-attention is disabled.")
        if context is not None:
            x = torch.cat([x, context], dim=1)
        return x

    def _build_time_embedding(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.time_embed(build_timestep_features(t, self.model_channels, flip_sin_to_cos=False, freq_shift=0))

    def _run_network(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context_ca: Optional[torch.Tensor],
        *,
        controlnet_residuals: dict | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = attention_mask, encoder_attention_mask
        x = self.pool(x)

        hs: list[torch.Tensor] = []
        h = x
        for block in self.input_blocks:
            h = block(h, emb, context_ca)
            hs.append(h)

        h = self.middle_block(h, emb, context_ca)

        if controlnet_residuals is not None:
            down_residuals = controlnet_residuals.get("down_residuals")
            if not isinstance(down_residuals, (list, tuple)):
                raise ValueError("controlnet_residuals['down_residuals'] must be a list/tuple of tensors.")
            if len(down_residuals) != len(hs):
                raise ValueError(
                    "controlnet down residual count mismatch: "
                    f"{len(down_residuals)} vs {len(hs)}"
                )
            hs = [base + residual for base, residual in zip(hs, down_residuals)]
            mid_residual = controlnet_residuals.get("mid_residual")
            if mid_residual is None:
                raise ValueError("controlnet_residuals missing 'mid_residual'.")
            h = h + mid_residual

        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb, context_ca)

        h = self.out(h)
        h = self.unpool(h)
        return h
