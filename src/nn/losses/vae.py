"""Backward-compatible VAE loss shim.

This module re-exports losses and discriminator helpers that were historically
implemented in this file. New code should import from the split modules in
`nn.losses` directly.
"""

from .adversarial import PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss
from .perceptual import PerceptualLoss
from .reconstruction import bce_focal_loss, focal_loss
from .regularization import latent_moment_regularizer, vq_regularizer

__all__ = [
    "PerceptualLoss",
    "PatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "latent_moment_regularizer",
    "vq_regularizer",
    "focal_loss",
    "bce_focal_loss",
]
