from __future__ import annotations

import torch

from models.vae.constants import LATENT_SCALE
from .base import BaseAutoencoder


def encode_to_latent(vae: BaseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Encode inputs into latent space using the framework VAE contract."""
    model_input = vae.image_to_model_range(x)
    posterior = vae.encode(model_input, normalize=False)
    if isinstance(posterior, torch.Tensor):
        return posterior
    return posterior.mode() * LATENT_SCALE


def decode_from_latent(
    vae: BaseAutoencoder,
    z: torch.Tensor,
    *,
    recon_type: str = "l1",
) -> torch.Tensor:
    """Decode latents into image space using the framework VAE contract."""
    raw = vae.decode(z, denorm=True)
    return vae.raw_output_to_image(raw, recon_type=recon_type)


__all__ = ["encode_to_latent", "decode_from_latent"]
