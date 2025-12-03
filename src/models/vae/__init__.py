"""
Variational Autoencoder building blocks and models.
"""

from .base import BaseVAE
from .kl import AutoencoderKL
from .vq import VQVAE
from .magvit import MagvitVQVAE

__all__ = [
    "BaseVAE",
    "AutoencoderKL",
    "VQVAE",
    "MagvitVQVAE",
]
