from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..registry import MODEL_REGISTRY
from models.unet.utils import TimestepEmbedding, build_timestep_features
from nn.blocks import BLOCK_REGISTRY
from nn.blocks.common import zero_module
from nn.ops.convolution import ConvND


@MODEL_REGISTRY.register("controlnet")
class ControlNetND(nn.Module):
    """Native ND ControlNet producing UNet-compatible residual tensors.

    Example:
        unet = UNet2DConditionND(...)
        controlnet = ControlNetND(...)
        residuals = controlnet(x, t, control_image, encoder_hidden_states=text_ctx)
        pred = unet(x, t, encoder_hidden_states=text_ctx, controlnet_residuals=residuals)
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        down_block_types: Sequence[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        block_out_channels: Sequence[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        dropout: float = 0.0,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        cross_attention_dim: int = 768,
        transformer_layers_per_block: int = 1,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
    ) -> None:
        super().__init__()
        self.block_out_channels = tuple(int(v) for v in block_out_channels)
        time_embed_dim = self.block_out_channels[0] * 4
        self.time_embedding_type = time_embedding_type
        self.freq_shift = freq_shift
        self.flip_sin_to_cos = flip_sin_to_cos
        self.time_proj_dim = self.block_out_channels[0]
        self.time_embedding = TimestepEmbedding(self.time_proj_dim, time_embed_dim)

        self.conv_in = ConvND(spatial_dims, in_channels, self.block_out_channels[0], kernel_size=3, padding=1).conv
        self.input_hint_block = nn.Sequential(
            ConvND(spatial_dims, conditioning_channels, self.block_out_channels[0], kernel_size=3, padding=1).conv,
            nn.SiLU(),
        )

        self.down_blocks = nn.ModuleList()
        self._residual_channels: list[int] = [self.block_out_channels[0]]

        output_channel = self.block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = self.block_out_channels[i]
            is_final = i == len(self.block_out_channels) - 1
            with_attention = down_block_type in {"AttnDownBlock2D", "CrossAttnDownBlock2D"}
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
                    with_attention=with_attention,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim if down_block_type == "CrossAttnDownBlock2D" else None,
                    transformer_layers_per_block=transformer_layers_per_block,
                )
            )
            self._residual_channels.extend([output_channel] * layers_per_block)
            if not is_final:
                self._residual_channels.append(output_channel)

        if mid_block_type is None:
            raise ValueError("ControlNetND requires a mid_block_type.")
        mid_block_cls = BLOCK_REGISTRY.get(mid_block_type)
        self.mid_block = mid_block_cls(
            spatial_dims=spatial_dims,
            in_channels=self.block_out_channels[-1],
            temb_channels=time_embed_dim,
            eps=norm_eps,
            groups=norm_num_groups,
            dropout=dropout,
            time_scale_shift=resnet_time_scale_shift,
            add_attention=True,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim if mid_block_type == "UNetMidBlock2DCrossAttn" else None,
            transformer_layers_per_block=transformer_layers_per_block,
        )

        self.zero_convs = nn.ModuleList(
            [zero_module(ConvND(spatial_dims, ch, ch, kernel_size=1, padding=0)) for ch in self._residual_channels]
        )
        self.mid_zero_conv = zero_module(
            ConvND(spatial_dims, self.block_out_channels[-1], self.block_out_channels[-1], kernel_size=1, padding=0)
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | float | int,
        controlnet_cond: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timesteps_emb: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        if timesteps_emb is None:
            timesteps = self._normalize_timesteps(timesteps, x)
            timesteps_emb = self._build_time_embedding(timesteps, x)

        sample = self.conv_in(x)
        hint = self.input_hint_block(controlnet_cond)
        if hint.shape != sample.shape:
            raise ValueError(
                f"controlnet_cond projected shape {tuple(hint.shape)} does not match sample {tuple(sample.shape)}."
            )
        sample = sample + hint

        down_block_res_samples: tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                sample,
                timesteps_emb,
                context=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            sample,
            timesteps_emb,
            context=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        down_residuals = [
            conv(tensor)
            for conv, tensor in zip(self.zero_convs, down_block_res_samples)
        ]
        mid_residual = self.mid_zero_conv(sample)
        return {"down_residuals": down_residuals, "mid_residual": mid_residual}

    @staticmethod
    def _normalize_timesteps(t, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        if t.ndim == 0:
            t = t[None].to(x.device)
        return t.expand(x.shape[0]).to(x.device)

    def _build_time_embedding(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.time_embedding_type != "positional":
            raise ValueError("ControlNetND currently supports positional time embedding only.")
        t_emb = build_timestep_features(
            t,
            self.time_proj_dim,
            max_period=10000,
            flip_sin_to_cos=self.flip_sin_to_cos,
            freq_shift=self.freq_shift,
        ).to(dtype=x.dtype)
        return self.time_embedding(t_emb)
