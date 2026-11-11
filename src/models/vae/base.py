"""
Abstract base for VAE models.
"""

from __future__ import annotations

import abc
import torch
import torch.nn as nn

from models.autoencoder import BaseAutoencoder
from .intermediate import (
    IntermediateFeaturePair,
    IntermediateLevelConfig,
    build_intermediate_alignment,
    normalize_intermediate_supervision,
)


class BaseVAE(BaseAutoencoder, metaclass=abc.ABCMeta):
    """Abstract base for VAEs."""

    def __init__(self) -> None:
        super().__init__()
        self.intermediate_alignments = nn.ModuleDict()
        self.intermediate_supervision: tuple[IntermediateLevelConfig, ...] = ()
        self._intermediate_supervision_enabled = False

    def configure_intermediate_supervision(self, raw_config) -> None:
        """Install opt-in feature aligners from the training configuration."""
        encoder_channels = tuple(int(value) for value in self.encoder.stage_channels)
        decoder_channels = tuple(int(value) for value in self.decoder.stage_channels)
        specs = normalize_intermediate_supervision(
            raw_config,
            stage_channels=encoder_channels,
            decoder_stage_channels=decoder_channels,
            spatial_dims=int(self.spatial_dims),
        )
        alignments = nn.ModuleDict()
        for spec in specs:
            alignments[str(spec.level)] = build_intermediate_alignment(
                spec.alignment,
                encoder_channels=encoder_channels[spec.level],
                decoder_channels=decoder_channels[spec.level],
                spatial_dims=int(self.spatial_dims),
            )
        self.intermediate_alignments = alignments
        self.intermediate_supervision = specs
        self._intermediate_supervision_enabled = bool(specs)

    def set_intermediate_supervision_enabled(self, enabled: bool) -> None:
        """Toggle activation capture without removing checkpointed projection modules."""
        self._intermediate_supervision_enabled = bool(enabled) and bool(self.intermediate_supervision)

    @property
    def intermediate_levels(self) -> tuple[int, ...]:
        if not self._intermediate_supervision_enabled:
            return ()
        return tuple(spec.level for spec in self.intermediate_supervision)

    def align_intermediate_features(
        self,
        encoder_features: dict[int, torch.Tensor],
        decoder_features: dict[int, torch.Tensor],
    ) -> dict[int, IntermediateFeaturePair] | None:
        if not self._intermediate_supervision_enabled or not self.intermediate_supervision:
            return None
        pairs: dict[int, IntermediateFeaturePair] = {}
        for spec in self.intermediate_supervision:
            level = spec.level
            if level not in encoder_features or level not in decoder_features:
                raise RuntimeError(f"Missing captured intermediate features for VAE level {level + 1}.")
            pairs[level] = self.intermediate_alignments[str(level)](
                encoder_features[level], decoder_features[level]
            )
        return pairs

    @abc.abstractmethod
    def make_discriminator(self):
        """Return a discriminator module."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, x: torch.Tensor, normalize: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, z: torch.Tensor, denorm: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        raise NotImplementedError
