"""
Variational Autoencoder building blocks and models.
"""

from core.types import ModelOutput
from . import monai_vae  # noqa: F401
from .constants import LATENT_SCALE
from .base import BaseVAE
from .kl import AutoencoderKL
from .vq import VQVAE
from .monai_vae import MonaiVAE
from .intermediate import (
    ConvPoolAlignment,
    IdentityAlignment,
    IntermediateAlignment,
    IntermediateAlignmentConfig,
    IntermediateFeaturePair,
    IntermediateLevelConfig,
    VectorProjectionAlignment,
    normalize_intermediate_supervision,
)

__all__ = [
    "BaseVAE",
    "LATENT_SCALE",
    "AutoencoderKL",
    "MonaiVAE",
    "VQVAE",
    "ModelOutput",
    "ConvPoolAlignment",
    "IdentityAlignment",
    "IntermediateAlignment",
    "IntermediateAlignmentConfig",
    "IntermediateFeaturePair",
    "IntermediateLevelConfig",
    "VectorProjectionAlignment",
    "normalize_intermediate_supervision",
]
