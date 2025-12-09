"""
Loss and regularization modules.
"""

from .vae import (
    PerceptualLoss,
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    vq_regularizer,
    focal_loss,
    bce_focal_loss,
)

__all__ = [
    "PerceptualLoss",
    "PatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "vq_regularizer",
    "focal_loss",
    "bce_focal_loss",
]
