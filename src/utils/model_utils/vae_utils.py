"""
Helpers for VAE model construction and inference.
"""

from __future__ import annotations

import torch

import utils
from models.factory import ModelFactory
from models.autoencoder.utils import encode_to_latent, reconstruct_from_image, sync_autoencoder_input_range


def build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
    """
    build_vae_model Function

    Builds a VAE model and optionally loads a checkpoint state.

    Inputs:
        - cfg: (dict) Full config dict.
        - device: (torch.device) Target device.
        - ckpt_path: (Path | None) Optional checkpoint path.
        - set_eval: (Boolean) If True, set model to eval() after loading.

    Outputs:
        - model: (torch.nn.Module) Constructed model.
    """
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    cfg_ckpt = model_cfg.get("ckpt_path")
    if isinstance(cfg_ckpt, str) and cfg_ckpt.lower() == "none":
        cfg_ckpt = None

    model = ModelFactory.build(cfg).to(device)
    sync_autoencoder_input_range(model, cfg)
    if ckpt_path is not None:
        payload = utils.safe_torch_load(ckpt_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    if set_eval:
        model.eval()
    return model


def encode_vae_batch(model, inputs: torch.Tensor, *, input_normalize: str = "centered") -> torch.Tensor:
    """
    encode_vae_batch Function

    Encodes inputs into latent representations.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in image space [0, 1].

    Outputs:
        - latents: (Tensor) Latent batch.
    """
    latents = encode_to_latent(model, inputs, input_normalize=input_normalize)
    return latents


def decode_vae_batch(model, latents: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
    """
    decode_vae_batch Function

    Decodes latent representations into images.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - latents: (Tensor) Latent batch.

    Outputs:
        - recon: (Tensor) Reconstructed images in image space [0, 1].
    """
    raw = model.decode(latents, denorm=False)
    return model.raw_output_to_image(raw, recon_type=recon_type)


def reconstruct_vae_batch(
    model,
    inputs: torch.Tensor,
    recon_type: str = "l1",
    *,
    input_normalize: str = "centered",
) -> torch.Tensor:
    """
    reconstruct_vae_batch Function

    Reconstructs inputs via encode+decode.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in image space [0, 1].

    Outputs:
        - recon: (Tensor) Reconstructed images in image space [0, 1].
    """
    return reconstruct_from_image(model, inputs, recon_type=recon_type, input_normalize=input_normalize)
