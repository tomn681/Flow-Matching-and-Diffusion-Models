"""
Autoencoder abstractions.
"""

from .base import BaseAutoencoder
from .utils import decode_from_latent, encode_to_latent

__all__ = ["BaseAutoencoder", "encode_to_latent", "decode_from_latent"]
