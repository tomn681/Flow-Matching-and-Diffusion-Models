from __future__ import annotations

import sys as _sys
import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..registry import MODEL_REGISTRY
from models.unet.base import BaseUNetND
from nn.ops import ConvND, ConvTransposeND
from nn.ops.time_embedding import timestep_embedding


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return positions.new_zeros((positions.numel(), 0))
    half = dim // 2
    if half == 0:
        return positions.new_zeros((positions.numel(), dim))
    freqs = torch.arange(half, device=positions.device, dtype=torch.float32)
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=positions.device)) * freqs / max(1, half - 1))
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, emb.new_zeros((emb.shape[0], dim - emb.shape[1]))], dim=1)
    return emb


def _factorized_nd_sincos_pos_embed(grid_shape: tuple[int, ...], dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("Position embedding dim must be > 0.")
    spatial_dims = len(grid_shape)
    if spatial_dims <= 0:
        raise ValueError("grid_shape must contain at least one spatial dimension.")
    base = dim // spatial_dims
    remainder = dim % spatial_dims
    axis_dims = [base + (1 if i < remainder else 0) for i in range(spatial_dims)]
    axis_dims = [d + (d % 2) for d in axis_dims]
    if sum(axis_dims) > dim:
        overflow = sum(axis_dims) - dim
        for i in range(len(axis_dims) - 1, -1, -1):
            shrink = min(overflow, axis_dims[i] % 2 + 2 if axis_dims[i] > 2 else axis_dims[i] % 2)
            if shrink <= 0:
                continue
            axis_dims[i] -= shrink
            overflow -= shrink
            if overflow <= 0:
                break
    axis_dims[-1] += dim - sum(axis_dims)
    meshes = torch.meshgrid(
        *[torch.arange(size, device=device, dtype=torch.float32) for size in grid_shape],
        indexing="ij",
    )
    parts = []
    for coords, axis_dim in zip(meshes, axis_dims):
        parts.append(_sincos_1d(coords.reshape(-1), axis_dim))
    pos = torch.cat(parts, dim=1)
    if pos.shape[1] != dim:
        if pos.shape[1] < dim:
            pos = torch.cat([pos, pos.new_zeros((pos.shape[0], dim - pos.shape[1]))], dim=1)
        else:
            pos = pos[:, :dim]
    return pos.to(dtype=dtype)


class AdaLNDiTBlock(nn.Module):
    """DiT block with adaLN-Zero modulation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        ff_mult = int(round(hidden_size * float(mlp_ratio)))
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ff_mult),
            nn.GELU(),
            nn.Linear(ff_mult, hidden_size),
        )
        self.modulation = nn.Linear(hidden_size, hidden_size * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(cond).chunk(6, dim=1)

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


@MODEL_REGISTRY.register("dit")
class DiTND(BaseUNetND):
    """ND patch-transformer denoiser with DiT-style adaLN-Zero conditioning."""

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
        use_adaLN: bool = True,
        zero_init_final_layer: bool = True,
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
        if learn_sigma:
            warnings.warn(
                "DiTND(learn_sigma=...) is deprecated. Set out_channels explicitly instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.learn_sigma = bool(learn_sigma)
        self.use_adaLN = bool(use_adaLN)
        self.zero_init_final_layer = bool(zero_init_final_layer)
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
        if self.zero_init_final_layer:
            nn.init.zeros_(self.unpatch.convT.weight)
            if getattr(self.unpatch.convT, "bias", None) is not None:
                nn.init.zeros_(self.unpatch.convT.bias)

        ff_mult = int(round(self.hidden_size * float(mlp_ratio)))
        if self.use_adaLN:
            self.blocks = nn.ModuleList(
                [
                    AdaLNDiTBlock(
                        hidden_size=self.hidden_size,
                        num_heads=int(num_heads),
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(depth))
                ]
            )
            self.final_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
            self.final_modulation = nn.Linear(self.hidden_size, self.hidden_size * 2)
            nn.init.zeros_(self.final_modulation.weight)
            nn.init.zeros_(self.final_modulation.bias)
        else:
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
            self.final_modulation = None
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
        controlnet_residuals: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = context_ca, controlnet_residuals, attention_mask, encoder_attention_mask
        try:
            bsz = x.shape[0]
            tokens = self.patch_embed(x)
            grid_shape = tokens.shape[2:]
            tokens = tokens.reshape(bsz, self.hidden_size, -1).transpose(1, 2).contiguous()

            pos = _factorized_nd_sincos_pos_embed(
                tuple(int(v) for v in grid_shape),
                self.hidden_size,
                device=x.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + pos.unsqueeze(0)

            cond = emb.to(dtype=tokens.dtype, device=tokens.device)
            if self.use_adaLN:
                for block in self.blocks:
                    tokens = block(tokens, cond)
                shift, scale = self.final_modulation(cond).chunk(2, dim=1)
                tokens = _modulate(self.final_norm(tokens), shift, scale)
            else:
                tokens = tokens + cond.unsqueeze(1)
                for block in self.blocks:
                    tokens = block(tokens)
                tokens = self.final_norm(tokens)

            tokens = tokens.transpose(1, 2).reshape(bsz, self.hidden_size, *grid_shape).contiguous()
            return self.unpatch(tokens)
        finally:
            self._current_y = None


PatchTransformerND = DiTND
try:
    MODEL_REGISTRY.register_value("patch_transformer", PatchTransformerND)
except ValueError:
    pass


_module = _sys.modules[__name__]
if __name__.startswith("genlib.models.dit."):
    _sys.modules.setdefault(__name__.replace("genlib.models.dit.", "models.dit.", 1), _module)
elif __name__.startswith("src.models.dit."):
    _sys.modules.setdefault(__name__.replace("src.models.dit.", "models.dit.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.models.dit.", "genlib.models.dit.", 1), _module)
elif __name__.startswith("models.dit."):
    _sys.modules.setdefault(__name__.replace("models.dit.", "src.models.dit.", 1), _module)
    _sys.modules.setdefault(__name__.replace("models.dit.", "genlib.models.dit.", 1), _module)
del _module, _sys
