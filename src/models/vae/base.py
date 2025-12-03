"""
Abstract base for VAE models.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseVAE(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base for VAEs."""

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
