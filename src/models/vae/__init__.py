"""
Variational Autoencoder building blocks and models.
Legacy `vae.py` is retained for backward compatibility; prefer the new modules.
"""

from . import vae as legacy  # deprecated
from .base import AutoencoderKL
from .vq import VQVAE
from .factory import VAEFactory, build_from_json
from .quantize import VectorQuantizerEMA
from .distributions import DiagonalGaussian

__all__ = [
    "AutoencoderKL",
    "VQVAE",
    "VAEFactory",
    "build_from_json",
    "VectorQuantizerEMA",
    "DiagonalGaussian",
    "legacy",
]
