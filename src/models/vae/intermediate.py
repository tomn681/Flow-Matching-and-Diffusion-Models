"""Typed contracts and projection modules for VAE intermediate supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from nn.ops.convolution import ConvND


@dataclass(frozen=True)
class IntermediateAlignmentConfig:
    """How an encoder/decoder feature pair is brought into a common space."""

    type: str = "identity"
    out_channels: int | None = None
    pool_size: int | tuple[int, ...] | None = None
    out_dim: int | None = None


@dataclass(frozen=True)
class IntermediateLevelConfig:
    """Normalized configuration for one active VAE stage."""

    level: int
    weight: float
    loss: str
    alignment: IntermediateAlignmentConfig


@dataclass
class IntermediateFeaturePair:
    """Aligned encoder and decoder representations for one logical VAE level."""

    encoder: torch.Tensor
    decoder: torch.Tensor


def _normalize_pool_size(value: Any, *, spatial_dims: int) -> tuple[int, ...]:
    if value is None:
        return (4,) * spatial_dims
    if isinstance(value, int):
        values = (int(value),) * spatial_dims
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(item) for item in value)
    else:
        raise TypeError("intermediate_supervision alignment.pool_size must be an int or sequence of ints.")
    if len(values) != spatial_dims or any(item <= 0 for item in values):
        raise ValueError(
            "intermediate_supervision alignment.pool_size must contain one positive value per spatial dimension."
        )
    return values


def _normalize_alignment(
    raw: Any,
    *,
    spatial_dims: int,
    encoder_channels: int,
    decoder_channels: int,
) -> IntermediateAlignmentConfig:
    if raw is None:
        data: dict[str, Any] = {"type": "identity"}
    elif isinstance(raw, str):
        data = {"type": raw}
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        raise TypeError("intermediate_supervision alignment must be a string or object.")

    alignment_type = str(data.get("type", "identity")).strip().lower()
    if alignment_type not in {"identity", "conv_pool", "vector_projection"}:
        raise ValueError(
            "intermediate_supervision alignment.type must be one of "
            "{'identity', 'conv_pool', 'vector_projection'}."
        )

    if alignment_type == "identity":
        unexpected = {key for key in ("out_channels", "pool_size", "out_dim") if data.get(key) is not None}
        if unexpected:
            raise ValueError(f"identity alignment does not accept: {', '.join(sorted(unexpected))}.")
        return IntermediateAlignmentConfig()

    if alignment_type == "conv_pool":
        out_channels = int(data.get("out_channels", min(encoder_channels, decoder_channels)))
        if out_channels <= 0:
            raise ValueError("conv_pool alignment.out_channels must be > 0.")
        return IntermediateAlignmentConfig(
            type=alignment_type,
            out_channels=out_channels,
            pool_size=_normalize_pool_size(data.get("pool_size"), spatial_dims=spatial_dims),
        )

    out_dim = int(data.get("out_dim", min(encoder_channels, decoder_channels)))
    if out_dim <= 0:
        raise ValueError("vector_projection alignment.out_dim must be > 0.")
    return IntermediateAlignmentConfig(type=alignment_type, out_dim=out_dim)


def normalize_intermediate_supervision(
    raw: Any,
    *,
    stage_channels: Sequence[int],
    decoder_stage_channels: Sequence[int],
    spatial_dims: int,
) -> tuple[IntermediateLevelConfig, ...]:
    """Normalize the compact JSON syntax into active, typed level configs."""

    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("training.intermediate_supervision must be an object or null.")

    levels = raw.get("levels")
    if not isinstance(levels, list):
        raise TypeError("training.intermediate_supervision.levels must be a list.")
    if len(levels) != len(stage_channels):
        raise ValueError(
            "training.intermediate_supervision.levels must have exactly one entry per VAE stage: "
            f"expected {len(stage_channels)}, got {len(levels)}."
        )
    if len(decoder_stage_channels) != len(stage_channels):
        raise ValueError("Encoder and decoder must expose the same number of stages for intermediate supervision.")

    default_loss = str(raw.get("default_loss", "l1")).strip().lower()
    default_alignment = raw.get("default_alignment", {"type": "identity"})
    supported_losses = {"l1", "mse", "smooth_l1", "cosine"}
    if default_loss not in supported_losses:
        raise ValueError(f"Unsupported intermediate default_loss '{default_loss}'.")

    normalized: list[IntermediateLevelConfig] = []
    for level, entry in enumerate(levels):
        if entry is None:
            continue
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            weight = float(entry)
            loss_name = default_loss
            alignment_raw = default_alignment
        elif isinstance(entry, Mapping):
            if "weight" not in entry:
                raise ValueError(f"intermediate_supervision level {level + 1} requires a weight.")
            weight = float(entry["weight"])
            loss_name = str(entry.get("loss", default_loss)).strip().lower()
            alignment_raw = entry.get("alignment", default_alignment)
        else:
            raise TypeError(
                f"intermediate_supervision level {level + 1} must be null, a numeric weight, or an object."
            )
        if not torch.isfinite(torch.tensor(weight)) or weight <= 0:
            raise ValueError(f"intermediate_supervision level {level + 1} weight must be finite and > 0.")
        if loss_name not in supported_losses:
            raise ValueError(
                f"Unsupported intermediate loss '{loss_name}' at level {level + 1}; "
                f"expected one of {sorted(supported_losses)}."
            )
        alignment = _normalize_alignment(
            alignment_raw,
            spatial_dims=spatial_dims,
            encoder_channels=int(stage_channels[level]),
            decoder_channels=int(decoder_stage_channels[level]),
        )
        normalized.append(
            IntermediateLevelConfig(level=level, weight=weight, loss=loss_name, alignment=alignment)
        )
    return tuple(normalized)


def _adaptive_avg_pool(spatial_dims: int, output_size: tuple[int, ...]) -> nn.Module:
    pool_cls = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}.get(spatial_dims)
    if pool_cls is None:
        raise ValueError("spatial_dims must be 1, 2, or 3.")
    return pool_cls(output_size)


class IntermediateAlignment(nn.Module):
    """Base strategy for aligning a pair of intermediate tensors."""

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> IntermediateFeaturePair:
        raise NotImplementedError


class IdentityAlignment(IntermediateAlignment):
    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> IntermediateFeaturePair:
        if encoder.shape != decoder.shape:
            raise ValueError(
                "identity intermediate alignment requires equal shapes, got "
                f"encoder={tuple(encoder.shape)} and decoder={tuple(decoder.shape)}."
            )
        return IntermediateFeaturePair(encoder=encoder, decoder=decoder)


class ConvPoolAlignment(IntermediateAlignment):
    """Lightweight 1x1 convolution followed by adaptive average pooling."""

    def __init__(
        self,
        *,
        encoder_channels: int,
        decoder_channels: int,
        out_channels: int,
        pool_size: tuple[int, ...],
        spatial_dims: int,
    ) -> None:
        super().__init__()
        self.encoder_projection = ConvND(spatial_dims, encoder_channels, out_channels, 1, padding=0)
        self.decoder_projection = ConvND(spatial_dims, decoder_channels, out_channels, 1, padding=0)
        self.pool = _adaptive_avg_pool(spatial_dims, pool_size)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> IntermediateFeaturePair:
        return IntermediateFeaturePair(
            encoder=self.pool(self.encoder_projection(encoder)),
            decoder=self.pool(self.decoder_projection(decoder)),
        )


class VectorProjectionAlignment(IntermediateAlignment):
    """Global average pooling followed by independent linear projections."""

    def __init__(
        self,
        *,
        encoder_channels: int,
        decoder_channels: int,
        out_dim: int,
        spatial_dims: int,
    ) -> None:
        super().__init__()
        self.pool = _adaptive_avg_pool(spatial_dims, (1,) * spatial_dims)
        self.encoder_projection = nn.Linear(encoder_channels, out_dim)
        self.decoder_projection = nn.Linear(decoder_channels, out_dim)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> IntermediateFeaturePair:
        encoder_vector = self.pool(encoder).flatten(1)
        decoder_vector = self.pool(decoder).flatten(1)
        return IntermediateFeaturePair(
            encoder=self.encoder_projection(encoder_vector),
            decoder=self.decoder_projection(decoder_vector),
        )


def build_intermediate_alignment(
    config: IntermediateAlignmentConfig,
    *,
    encoder_channels: int,
    decoder_channels: int,
    spatial_dims: int,
) -> IntermediateAlignment:
    if config.type == "identity":
        return IdentityAlignment()
    if config.type == "conv_pool":
        assert config.out_channels is not None and isinstance(config.pool_size, tuple)
        return ConvPoolAlignment(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            out_channels=config.out_channels,
            pool_size=config.pool_size,
            spatial_dims=spatial_dims,
        )
    if config.type == "vector_projection":
        assert config.out_dim is not None
        return VectorProjectionAlignment(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            out_dim=config.out_dim,
            spatial_dims=spatial_dims,
        )
    raise ValueError(f"Unsupported intermediate alignment type '{config.type}'.")


__all__ = [
    "ConvPoolAlignment",
    "IdentityAlignment",
    "IntermediateAlignment",
    "IntermediateAlignmentConfig",
    "IntermediateFeaturePair",
    "IntermediateLevelConfig",
    "VectorProjectionAlignment",
    "build_intermediate_alignment",
    "normalize_intermediate_supervision",
]
