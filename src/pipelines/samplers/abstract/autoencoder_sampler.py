"""
Abstract interface for autoencoder-family samplers.
"""

from __future__ import annotations

from .sampler import AbstractSampler


class AbstractAutoencoderSampler(AbstractSampler):
    """
    Marker interface for autoencoder samplers (VAE and future autoencoders).
    """

    pass

