"""
Diffusers-like UNet implemented with ND operators.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.unet.base import BaseUNetND
from models.unet.utils import GaussianFourierProjection, Timesteps, TimestepEmbedding, build_timestep_features
from ..registry import MODEL_REGISTRY
from nn.blocks import BLOCK_REGISTRY
from nn.ops.convolution import ConvND
from nn.ops.normalization import make_group_norm


@MODEL_REGISTRY.register("diffusers_unet")
class UNetDiffusersND(BaseUNetND):
    """
    Diffusers-compat UNet for ND use.

    For strict legacy 2D checkpoint conversion, keep spatial_dims=2 and use
    the same config as the source UNet2DModel.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        sample_size: int | Sequence[int] | None = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: int | None = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Sequence[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        mid_block_type: str | None = "UNetMidBlock2D",
        up_block_types: Sequence[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Sequence[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: int | None = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: str | None = None,
        num_class_embeds: int | None = None,
        num_train_timesteps: int | None = None,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
        **_kwargs,
    ):
        super().__init__()
        self.center_input_sample = center_input_sample
        self.sample_size = sample_size
        self.time_embedding_type = time_embedding_type
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
        self.block_out_channels = tuple(block_out_channels)
        self.cross_attention_dim = int(cross_attention_dim) if cross_attention_dim is not None else None

        time_embed_dim = time_embedding_dim or (self.block_out_channels[0] * 4)
        self.conv_in = ConvND(spatial_dims, in_channels, self.block_out_channels[0], kernel_size=3, padding=1).conv

        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=self.block_out_channels[0],
                scale=16,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            self.time_proj_dim = 2 * self.block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(self.block_out_channels[0], flip_sin_to_cos, freq_shift)
            self.time_proj_dim = self.block_out_channels[0]
        elif time_embedding_type == "learned":
            if num_train_timesteps is None:
                raise ValueError("UNetDiffusersND learned time embeddings require num_train_timesteps.")
            self.time_proj = nn.Embedding(num_train_timesteps, self.block_out_channels[0])
            self.time_proj_dim = self.block_out_channels[0]
        else:
            raise ValueError(f"Unsupported time_embedding_type '{time_embedding_type}'")
        self.time_embedding = TimestepEmbedding(self.time_proj_dim, time_embed_dim, act_fn=act_fn)
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(self.time_proj_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity()
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        output_channel = self.block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = self.block_out_channels[i]
            is_final = i == len(self.block_out_channels) - 1
            with_attention = down_block_type in {"AttnDownBlock2D", "CrossAttnDownBlock2D"}
            if down_block_type not in {"DownBlock2D", "AttnDownBlock2D", "CrossAttnDownBlock2D"}:
                raise ValueError(f"Unsupported down block type in compat model: {down_block_type}")
            block_cls = BLOCK_REGISTRY.get(down_block_type)
            self.down_blocks.append(
                block_cls(
                    spatial_dims=spatial_dims,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final,
                    eps=norm_eps,
                    groups=norm_num_groups,
                    dropout=dropout,
                    time_scale_shift=resnet_time_scale_shift,
                    act_fn=act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                    with_attention=with_attention,
                    attention_head_dim=attention_head_dim,
                    attn_norm_num_groups=attn_norm_num_groups,
                    downsample_padding=downsample_padding,
                    downsample_type=downsample_type,
                    cross_attention_dim=self.cross_attention_dim if down_block_type == "CrossAttnDownBlock2D" else None,
                    transformer_layers_per_block=transformer_layers_per_block,
                )
            )

        if mid_block_type is None:
            self.mid_block = None
        else:
            mid_block_cls = BLOCK_REGISTRY.get(mid_block_type)
            self.mid_block = mid_block_cls(
                spatial_dims=spatial_dims,
                in_channels=self.block_out_channels[-1],
                temb_channels=time_embed_dim,
                eps=norm_eps,
                groups=norm_num_groups,
                dropout=dropout,
                time_scale_shift=resnet_time_scale_shift,
                act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                attn_groups=attn_norm_num_groups,
                pre_norm=True,
                add_attention=add_attention,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=self.cross_attention_dim
                if mid_block_type == "UNetMidBlock2DCrossAttn"
                else None,
                transformer_layers_per_block=transformer_layers_per_block,
            )

        reversed_channels = list(reversed(self.block_out_channels))
        output_channel = reversed_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, len(self.block_out_channels) - 1)]
            is_final = i == len(self.block_out_channels) - 1
            with_attention = up_block_type in {"AttnUpBlock2D", "CrossAttnUpBlock2D"}
            if up_block_type not in {"UpBlock2D", "AttnUpBlock2D", "CrossAttnUpBlock2D"}:
                raise ValueError(f"Unsupported up block type in compat model: {up_block_type}")
            block_cls = BLOCK_REGISTRY.get(up_block_type)
            self.up_blocks.append(
                block_cls(
                    spatial_dims=spatial_dims,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final,
                    eps=norm_eps,
                    groups=norm_num_groups,
                    dropout=dropout,
                    time_scale_shift=resnet_time_scale_shift,
                    act_fn=act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                    with_attention=with_attention,
                    attention_head_dim=attention_head_dim,
                    attn_norm_num_groups=attn_norm_num_groups,
                    upsample_type=upsample_type,
                    cross_attention_dim=self.cross_attention_dim if up_block_type == "CrossAttnUpBlock2D" else None,
                    transformer_layers_per_block=transformer_layers_per_block,
                )
            )

        self.conv_norm_out = make_group_norm(self.block_out_channels[0], groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = ConvND(spatial_dims, self.block_out_channels[0], out_channels, kernel_size=3, padding=1).conv

    def _prepare_input(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        context_ca: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is not None:
            x = torch.cat([x, context], dim=1)
        if self.center_input_sample:
            x = 2 * x - 1.0
        return x

    def _build_time_embedding(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.time_embedding_type == "learned":
            t_emb = self.time_proj(t)
        else:
            t_emb = self.time_proj(t).to(dtype=x.dtype)
        return self.time_embedding(t_emb)

    def _run_network(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context_ca: torch.Tensor | None,
        controlnet_residuals: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = self.conv_in(x)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                sample,
                emb,
                context=context_ca,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            down_block_res_samples += res_samples

        if controlnet_residuals is not None:
            down_residuals = controlnet_residuals.get("down_residuals")
            if not isinstance(down_residuals, (list, tuple)):
                raise ValueError("controlnet_residuals['down_residuals'] must be a list/tuple of tensors.")
            if len(down_residuals) != len(down_block_res_samples):
                raise ValueError(
                    "controlnet down residual count mismatch: "
                    f"{len(down_residuals)} vs {len(down_block_res_samples)}"
                )
            down_block_res_samples = tuple(
                base + residual for base, residual in zip(down_block_res_samples, down_residuals)
            )

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                context=context_ca,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            if controlnet_residuals is not None:
                mid_residual = controlnet_residuals.get("mid_residual")
                if mid_residual is None:
                    raise ValueError("controlnet_residuals missing 'mid_residual'.")
                sample = sample + mid_residual

        for upsample_block in self.up_blocks:
            n_res = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-n_res:]
            down_block_res_samples = down_block_res_samples[:-n_res]
            sample = upsample_block(
                sample,
                res_samples,
                emb,
                context=context_ca,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        context: torch.Tensor | None = None,
        context_ca: torch.Tensor | None = None,
        controlnet_residuals: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self._prepare_input(x, context, context_ca)
        t = self._normalize_timesteps(t, x)
        emb = self._build_time_embedding(t, x)
        if class_labels is not None:
            if self.class_embedding is None:
                raise ValueError("UNetDiffusersND received class_labels, but class embedding is not configured.")
            if isinstance(self.class_embedding, TimestepEmbedding):
                class_proj = self.time_proj(class_labels).to(dtype=x.dtype)
                emb = emb + self.class_embedding(class_proj)
            else:
                emb = emb + self.class_embedding(class_labels)
        y = self._run_network(
            x,
            emb,
            context_ca,
            controlnet_residuals=controlnet_residuals,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self._postprocess_output(y)


# Backward-compatible alias while configs/imports migrate.
UNetExactND = UNetDiffusersND
