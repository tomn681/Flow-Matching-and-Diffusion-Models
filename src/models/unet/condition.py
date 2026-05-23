"""
UNet2DConditionModel-style ND UNet.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .diffusers import UNetDiffusersND
from ..registry import MODEL_REGISTRY
from models.unet.utils import build_timestep_features


@MODEL_REGISTRY.register("condition_unet")
class UNet2DConditionND(UNetDiffusersND):
    """
    Diffusers UNet2DConditionModel-style UNet extended to ND.

    Supports encoder_hidden_states cross-attention plus optional class and
    timestep-condition embeddings.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        sample_size: int | Sequence[int] | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Sequence[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        up_block_types: Sequence[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Sequence[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1.0,
        dropout: float = 0.0,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        cross_attention_dim: int = 1280,
        class_embed_type: str | None = None,
        num_class_embeds: int | None = None,
        time_cond_proj_dim: int | None = None,
        transformer_layers_per_block: int = 1,
        **kwargs,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            dropout=dropout,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            **kwargs,
        )
        self.class_embed_type = class_embed_type
        self.time_cond_proj_dim = time_cond_proj_dim
        temb_dim = self.block_out_channels[0] * 4

        if class_embed_type is None:
            self.class_embedding = None
        elif class_embed_type == "timestep":
            self.class_embedding = nn.Sequential(
                nn.Linear(self.time_proj_dim, temb_dim),
                nn.SiLU(),
                nn.Linear(temb_dim, temb_dim),
            )
        elif class_embed_type in {"identity", "projection"}:
            if num_class_embeds is None:
                raise ValueError("num_class_embeds is required for class embedding")
            self.class_embedding = nn.Embedding(num_class_embeds, temb_dim)
        else:
            raise ValueError(f"Unsupported class_embed_type '{class_embed_type}'.")

        self.time_cond_proj = (
            nn.Linear(time_cond_proj_dim, temb_dim) if time_cond_proj_dim is not None else None
        )

    def _class_embedding(self, class_labels: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if self.class_embedding is None or class_labels is None:
            return None
        if self.class_embed_type == "timestep":
            class_labels = class_labels.to(x.device)
            if class_labels.ndim == 0:
                class_labels = class_labels[None]
            class_labels = class_labels.expand(x.shape[0])
            class_t = build_timestep_features(
                class_labels,
                self.time_proj_dim,
                max_period=10000,
                flip_sin_to_cos=self.flip_sin_to_cos,
                freq_shift=self.freq_shift,
            ).to(dtype=x.dtype)
            return self.class_embedding(class_t)
        return self.class_embedding(class_labels.to(x.device).long())

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_ca: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # keep compat with existing call sites: prefer explicit encoder_hidden_states,
        # fallback to context_ca used by current trainer.
        if encoder_hidden_states is None:
            encoder_hidden_states = context_ca

        x = self._prepare_input(x, context, encoder_hidden_states)
        t = self._normalize_timesteps(t, x)
        emb = self._build_time_embedding(t, x)

        class_emb = self._class_embedding(class_labels, x)
        if class_emb is not None:
            emb = emb + class_emb
        if self.time_cond_proj is not None and timestep_cond is not None:
            emb = emb + self.time_cond_proj(timestep_cond.to(device=x.device, dtype=emb.dtype))

        y = self._run_network(
            x,
            emb,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self._postprocess_output(y)
