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

    input_range: str = "minus_one_to_one"

    def _normalized_input_range(self) -> str:
        mode = str(getattr(self, "input_range", "minus_one_to_one")).lower()
        aliases = {
            "minus_one_to_one": "minus_one_to_one",
            "-1,1": "minus_one_to_one",
            "neg1_to_1": "minus_one_to_one",
            "zero_to_one": "zero_to_one",
            "0,1": "zero_to_one",
        }
        normalized = aliases.get(mode)
        if normalized is None:
            raise ValueError(
                f"Unsupported autoencoder input_range '{mode}'. "
                "Expected 'minus_one_to_one' or 'zero_to_one'."
            )
        return normalized

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalized_input_range() == "zero_to_one":
            return x
        return x * 2.0 - 1.0

    def model_to_image_range(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalized_input_range() == "zero_to_one":
            return x.clamp(0.0, 1.0)
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
