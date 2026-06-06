from __future__ import annotations

import torch

from models.vae.constants import LATENT_SCALE
from .base import BaseAutoencoder


def resolve_input_normalize(vae: BaseAutoencoder, mode: str | None = None) -> str:
    """Resolve trainer/sampler normalization mode with backward-compatible model fallback."""
    if mode is not None:
        return str(mode).lower()
    input_range = str(getattr(vae, "input_range", "minus_one_to_one")).lower()
    if input_range in {"zero_to_one", "0,1"}:
        return "positive"
    return "centered"


def apply_input_normalize(x: torch.Tensor, mode: str = "centered") -> torch.Tensor:
    """Normalize image-space inputs before VAE encoding.

    Args:
        x: Input tensor in canonical image space, typically [0, 1].
        mode: One of {"centered", "positive", "zscore"}.
    """
    normalized = str(mode).lower()
    if normalized == "centered":
        return x * 2.0 - 1.0
    if normalized == "positive":
        return x
    if normalized == "zscore":
        reduce_dims = tuple(range(1, x.ndim))
        mu = x.mean(dim=reduce_dims, keepdim=True)
        sigma = x.std(dim=reduce_dims, keepdim=True).clamp(min=1e-6)
        return (x - mu) / sigma
    raise ValueError(
        f"Unknown input_normalize mode '{mode}'. "
        "Expected one of: 'centered', 'positive', 'zscore'."
    )


def encode_to_latent(
    vae: BaseAutoencoder,
    x: torch.Tensor,
    *,
    input_normalize: str | None = None,
) -> torch.Tensor:
    """Encode inputs into latent space using the framework VAE contract."""
    model_input = apply_input_normalize(x, resolve_input_normalize(vae, input_normalize))
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


def reconstruct_from_image(
    vae: BaseAutoencoder,
    x: torch.Tensor,
    *,
    recon_type: str = "l1",
    input_normalize: str | None = None,
) -> torch.Tensor:
    """Encode+decode from canonical image space using an explicit input normalization mode."""
    model_input = apply_input_normalize(x, resolve_input_normalize(vae, input_normalize))
    outputs = vae(model_input, sample_posterior=False)
    if hasattr(outputs, "reconstruction"):
        recon = outputs.reconstruction
    elif isinstance(outputs, (list, tuple)):
        recon = outputs[0]
    else:
        recon = outputs
    return vae.raw_output_to_image(recon, recon_type=recon_type)


__all__ = [
    "resolve_input_normalize",
    "apply_input_normalize",
    "encode_to_latent",
    "decode_from_latent",
    "reconstruct_from_image",
]
