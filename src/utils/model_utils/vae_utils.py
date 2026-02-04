"""
Helpers for VAE model construction and inference.
"""

from __future__ import annotations

import torch

from models.generators.vaefactory import VAEFactory


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
    cfg_path = cfg.get("__config_path__")
    if not cfg_path:
        raise ValueError("Missing __config_path__ in config.")
    model = VAEFactory().build_from_json(cfg_path).to(device)
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    if set_eval:
        model.eval()
    return model


def encode_vae_batch(model, inputs: torch.Tensor) -> torch.Tensor:
    """
    encode_vae_batch Function

    Encodes inputs into latent representations.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in [-1, 1].

    Outputs:
        - latents: (Tensor) Latent batch.
    """
    posterior = model.encode(inputs, normalize=False)
    return posterior.mode()


def decode_vae_batch(model, latents: torch.Tensor) -> torch.Tensor:
    """
    decode_vae_batch Function

    Decodes latent representations into images.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - latents: (Tensor) Latent batch.

    Outputs:
        - recon: (Tensor) Reconstructed images.
    """
    return model.decode(latents, denorm=False)


def reconstruct_vae_batch(model, inputs: torch.Tensor) -> torch.Tensor:
    """
    reconstruct_vae_batch Function

    Reconstructs inputs via encode+decode.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in [-1, 1].

    Outputs:
        - recon: (Tensor) Reconstructed images.
    """
    outputs = model(inputs, sample_posterior=False)
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs
