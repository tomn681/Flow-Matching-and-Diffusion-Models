"""
MAGVIT-style VQ-VAE with custom discriminator.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .vq import VQVAE
from nn.modules.vae.discriminators import MagvitDiscriminator


class MagvitVQVAE(VQVAE):
    """
    MAGVIT-inspired VQ-VAE.
    Uses the same core architecture as VQVAE but overrides the discriminator.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def make_discriminator(self):
        return MagvitDiscriminator(in_channels=self.decoder.conv_out.out_channels)
