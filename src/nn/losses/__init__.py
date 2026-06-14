"""
Tensor-level loss math and small discriminator modules.

Boundary:
- `src.nn.losses` owns raw tensor operations and local NN primitives.
- `src.losses` wraps these primitives into trainer-facing registered components.
"""

from .adversarial import PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss
from .perceptual import PerceptualLoss
from .reconstruction import bce_focal_loss, focal_loss
from .regularization import latent_moment_regularizer, vq_regularizer
from .ssim import ssim_loss

__all__ = [
    "PerceptualLoss",
    "PatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "latent_moment_regularizer",
    "vq_regularizer",
    "focal_loss",
    "bce_focal_loss",
    "ssim_loss",
]
