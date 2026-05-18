"""
Loss and regularization modules.
"""

from .adversarial import PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss
from .perceptual import PerceptualLoss
from .reconstruction import bce_focal_loss, focal_loss
from .regularization import vq_regularizer

__all__ = [
    "PerceptualLoss",
    "PatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "vq_regularizer",
    "focal_loss",
    "bce_focal_loss",
]
