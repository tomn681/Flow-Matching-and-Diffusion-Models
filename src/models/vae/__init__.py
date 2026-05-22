"""
Variational Autoencoder building blocks and models.
"""

from core.types import ModelOutput
from .constants import LATENT_SCALE
from .base import BaseVAE
from .kl import AutoencoderKL
from .vq import VQVAE

__all__ = [
    "BaseVAE",
    "LATENT_SCALE",
    "AutoencoderKL",
    "VQVAE",
    "ModelOutput",
]
