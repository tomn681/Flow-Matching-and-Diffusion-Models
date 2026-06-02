from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..registry import MODEL_REGISTRY
from models.unet.base import BaseUNetND
from nn.ops import ConvND, ConvTransposeND
from nn.ops.time_embedding import timestep_embedding


@MODEL_REGISTRY.register("dit")
class DiTND(BaseUNetND):
    """Minimal ND Diffusion Transformer denoiser.

    This implementation uses additive token conditioning rather than the paper's
    adaLN modulation. The tradeoff is architectural simplicity in exchange for a
    less expressive conditioning path.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 4,
        out_channels: int | None = None,
        patch_size: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int | None = None,
        class_dropout_prob: float = 0.0,
        learn_sigma: bool = False,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        if depth <= 0:
            raise ValueError("depth must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.spatial_dims = int(spatial_dims)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.learn_sigma = bool(learn_sigma)
        if out_channels is None:
            out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.out_channels = int(out_channels)

        kernel = (self.patch_size,) * self.spatial_dims
        self.patch_embed = ConvND(
            self.spatial_dims,
            self.in_channels,
            self.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
        )
        self.unpatch = ConvTransposeND(
            self.spatial_dims,
            self.hidden_size,
            self.out_channels,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
        )

        ff_mult = int(round(self.hidden_size * float(mlp_ratio)))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=int(num_heads),
                    dim_feedforward=ff_mult,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.class_embed = None
        self.class_dropout_prob = float(class_dropout_prob)
        if num_classes is not None:
            self.class_embed = nn.Embedding(int(num_classes), self.hidden_size)
        self._current_y: Optional[torch.Tensor] = None

    def _validate_shape(self, x: torch.Tensor) -> None:
        if x.dim() != self.spatial_dims + 2:
            raise ValueError(
                f"Expected rank {self.spatial_dims + 2} input for spatial_dims={self.spatial_dims}, got {x.dim()}."
            )
        spatial = x.shape[2:]
        if any(int(size) % self.patch_size != 0 for size in spatial):
            raise ValueError(
                f"All spatial dimensions must be divisible by patch_size={self.patch_size}. Got {tuple(spatial)}."
            )

    def _time_condition(self, t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        t = t.to(device=device)
        t_features = timestep_embedding(timesteps=t, dim=self.hidden_size).to(dtype=dtype)
        return self.time_mlp(t_features)

    def _prepare_input(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        context_ca: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = context_ca
        self._validate_shape(x)
        self._current_y = None
        if context is not None:
            if not torch.is_tensor(context):
                raise TypeError("DiTND class conditioning context must be a tensor when provided.")
            if context.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.long,
            }:
                raise TypeError("DiTND only supports integer class-label conditioning via context.")
            labels = context.reshape(-1)
            if labels.shape[0] != x.shape[0]:
                raise ValueError("DiTND class-label conditioning batch size must match the input batch size.")
            self._current_y = labels.to(device=x.device, dtype=torch.long)
        return x

    def _build_time_embedding(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cond = self._time_condition(t, dtype=x.dtype, device=x.device)
        if self.class_embed is not None and self._current_y is not None:
            class_cond = self.class_embed(self._current_y).to(dtype=cond.dtype)
            if self.training and self.class_dropout_prob > 0.0:
                keep = (torch.rand(class_cond.shape[0], device=class_cond.device) >= self.class_dropout_prob).to(
                    class_cond.dtype
                )
                class_cond = class_cond * keep.unsqueeze(1)
            cond = cond + class_cond
        return cond

    def _run_network(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context_ca: Optional[torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = context_ca, attention_mask, encoder_attention_mask
        try:
            bsz = x.shape[0]
            tokens = self.patch_embed(x)
            grid_shape = tokens.shape[2:]
            tokens = tokens.reshape(bsz, self.hidden_size, -1).transpose(1, 2).contiguous()

            token_count = tokens.shape[1]
            pos = timestep_embedding(
                timesteps=torch.arange(token_count, device=x.device),
                dim=self.hidden_size,
            ).to(dtype=tokens.dtype)
            tokens = tokens + pos.unsqueeze(0)

            tokens = tokens + emb.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(1)

            for block in self.blocks:
                tokens = block(tokens)
            tokens = self.final_norm(tokens)

            tokens = tokens.transpose(1, 2).reshape(bsz, self.hidden_size, *grid_shape).contiguous()
            return self.unpatch(tokens)
        finally:
            self._current_y = None
