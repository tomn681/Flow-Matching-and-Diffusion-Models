"""
Helpers for VAE model construction and inference.
"""

from __future__ import annotations

import warnings

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
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    cfg_ckpt = model_cfg.get("ckpt_path")
    if isinstance(cfg_ckpt, str) and cfg_ckpt.lower() == "none":
        cfg_ckpt = None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*No checkpoint provided\. Random initialization\.",
        )
        model = VAEFactory().build_from_json(cfg_path).to(device)
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    if set_eval:
        model.eval()
        if ckpt_path is None and cfg_ckpt is None:
            warnings.warn("[VAE] No checkpoint provided. Random initialization.")
    return model


def encode_vae_batch(model, inputs: torch.Tensor) -> torch.Tensor:
    """
    encode_vae_batch Function

    Encodes inputs into latent representations.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in image space [0, 1].

    Outputs:
        - latents: (Tensor) Latent batch.
    """
    posterior = model.encode(model.image_to_model_range(inputs), normalize=False)
    return posterior.mode()


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


def reconstruct_vae_batch(model, inputs: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
    """
    reconstruct_vae_batch Function

    Reconstructs inputs via encode+decode.

    Inputs:
        - model: (torch.nn.Module) VAE model.
        - inputs: (Tensor) Input batch in image space [0, 1].

    Outputs:
        - recon: (Tensor) Reconstructed images in image space [0, 1].
    """
    model_inputs = model.image_to_model_range(inputs)
    outputs = model(model_inputs, sample_posterior=False)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    return model.raw_output_to_image(outputs, recon_type=recon_type)


def run_self_tests() -> None:
    """
    Lightweight tests for VAE utility helpers.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch unavailable
        raise RuntimeError("torch is required for VAE utility self-tests.") from exc

    class DummyPosterior:
        def __init__(self, x):
            self._x = x

        def mode(self):
            return self._x + 1.0

    class DummyModel:
        def image_to_model_range(self, x):
            return x * 2.0 - 1.0

        def raw_output_to_image(self, x, recon_type="l1"):
            return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

        def encode(self, x, normalize=False):
            return DummyPosterior(x)

        def decode(self, z, denorm=False):
            return z - 1.0

        def __call__(self, x, sample_posterior=False):
            return x * 0.5

    model = DummyModel()
    inputs = torch.zeros(2, 1, 2, 2)
    latents = encode_vae_batch(model, inputs)
    recon = decode_vae_batch(model, latents)
    full = reconstruct_vae_batch(model, inputs)
    assert latents.shape == inputs.shape
    assert recon.shape == inputs.shape
    assert full.shape == inputs.shape
