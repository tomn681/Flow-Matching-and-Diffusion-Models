"""
Abstract base for VAE models.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseVAE(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base for VAEs."""

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        """Map image-domain tensors from [0, 1] to the model working range [-1, 1]."""
        return x * 2.0 - 1.0

    def model_to_image_range(self, x: torch.Tensor) -> torch.Tensor:
        """Map model-space image tensors from [-1, 1] to image space [0, 1]."""
        return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        """
        Convert a raw decoder output into image-domain [0, 1] tensors.

        BCE-style reconstruction treats the decoder output as logits; other
        reconstruction modes treat it as model-space image values.
        """
        recon_key = str(recon_type).lower()
        if recon_key in {"bce", "focal", "bce_focal"}:
            return torch.sigmoid(x)
        return self.model_to_image_range(x)

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
