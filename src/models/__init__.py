"""
Model architectures assembled from the core building blocks.

`ModelFactory` is the unified entrypoint for model construction.
"""

from . import adapters, autoencoder, controlnet, dit, unet, vae
from core.types import ModelOutput
from .autoencoder.base import BaseAutoencoder
from .factory import ModelFactory
from .registry import MODEL_REGISTRY
from .vae.base import BaseVAE
from .vae.kl import AutoencoderKL
from .vae.vq import VQVAE
from .generators import VAEFactory, build_from_json
from .controlnet import ControlNetND
from .dit import DiTND

__all__ = [
    "autoencoder",
    "adapters",
    "unet",
    "dit",
    "controlnet",
    "vae",
    "BaseAutoencoder",
    "BaseVAE",
    "ModelFactory",
    "MODEL_REGISTRY",
    "AutoencoderKL",
    "VQVAE",
    "ModelOutput",
    "VAEFactory",
    "build_from_json",
    "ControlNetND",
    "DiTND",
]
