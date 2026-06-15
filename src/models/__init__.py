"""
Model architectures assembled from the core building blocks.

`ModelFactory` is the unified entrypoint for model construction.
"""

from __future__ import annotations

from importlib import import_module

from core.types import ModelOutput
from .autoencoder.base import BaseAutoencoder
from .factory import ModelFactory
from .generators import VAEFactory, build_from_json
from .registry import MODEL_REGISTRY
from .utils import merge_models
from .vae.base import BaseVAE
from .vae.kl import AutoencoderKL
from .vae.vq import VQVAE

_LAZY_EXPORTS = {
    "adapters": ("models.adapters", None),
    "autoencoder": ("models.autoencoder", None),
    "controlnet": ("models.controlnet", None),
    "dit": ("models.dit", None),
    "unet": ("models.unet", None),
    "utils": ("models.utils", None),
    "vae": ("models.vae", None),
    "ControlNetND": ("models.controlnet", "ControlNetND"),
    "DiTND": ("models.dit", "DiTND"),
    "VideoUNetND": ("models.unet", "VideoUNetND"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return module if attr is None else getattr(module, attr)


__all__ = [
    "autoencoder",
    "adapters",
    "unet",
    "dit",
    "controlnet",
    "utils",
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
    "VideoUNetND",
    "merge_models",
]
