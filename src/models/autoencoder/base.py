"""
Abstract base for autoencoder-family models.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module, metaclass=abc.ABCMeta):
    """
    Base contract for image autoencoders.
    """

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    def model_to_image_range(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        recon_key = str(recon_type).lower()
        if recon_key in {"bce", "focal", "bce_focal"}:
            return torch.sigmoid(x)
        return self.model_to_image_range(x)

    @abc.abstractmethod
    def encode(self, x: torch.Tensor, normalize: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, z: torch.Tensor, denorm: bool = False):
        raise NotImplementedError

