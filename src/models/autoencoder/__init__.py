"""
Autoencoder abstractions.
"""

from .base import BaseAutoencoder
from .utils import encode_to_latent

__all__ = ["BaseAutoencoder", "encode_to_latent"]
