from __future__ import annotations

import logging

import torch

from models.vae.constants import LATENT_SCALE
from .base import BaseAutoencoder


def resolve_input_normalize(vae: BaseAutoencoder, mode: str | None = None) -> str:
    """Resolve trainer/sampler normalization mode with backward-compatible model fallback."""
    if mode is not None:
        normalized = str(mode).lower()
        if normalized == "symmetric":
            return "centered"
        return normalized
    input_range = str(getattr(vae, "input_range", "minus_one_to_one")).lower()
    if input_range in {"zero_to_one", "0,1"}:
        return "positive"
    return "centered"


def resolve_model_input_range_from_normalize(mode: str | None) -> str | None:
    """Derive model input/output range from trainer-side normalization when possible."""
    if mode is None:
        return None
    normalized = str(mode).lower()
    if normalized == "positive":
        return "zero_to_one"
    if normalized in {"centered", "symmetric"}:
        return "minus_one_to_one"
    return None


def resolve_latent_scaling_factor(vae: BaseAutoencoder) -> float:
    value = getattr(vae, "scaling_factor", LATENT_SCALE)
    try:
        return float(value)
    except Exception:
        return float(LATENT_SCALE)


def extract_autoencoder_contract(
    vae: BaseAutoencoder,
    cfg: dict | None = None,
    *,
    input_normalize: str | None = None,
) -> dict[str, object]:
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    normalize_mode = input_normalize
    if normalize_mode is None and isinstance(training_cfg, dict):
        normalize_mode = training_cfg.get("input_normalize")
    resolved_normalize = resolve_input_normalize(vae, normalize_mode)
    input_range = str(getattr(vae, "input_range", "minus_one_to_one"))
    return {
        "input_normalize": resolved_normalize,
        "input_range": input_range,
        "data_range": "zero_to_one" if resolved_normalize == "positive" else input_range,
        "latent_norm": training_cfg.get("latent_norm") if isinstance(training_cfg, dict) else None,
        "scaling_factor": resolve_latent_scaling_factor(vae),
    }


def apply_autoencoder_checkpoint_contract(
    vae: BaseAutoencoder,
    payload: dict | None,
    cfg: dict | None = None,
    *,
    input_normalize: str | None = None,
) -> dict[str, object] | None:
    sync_autoencoder_input_range(vae, cfg, input_normalize=input_normalize)
    contract = None
    if isinstance(payload, dict):
        contract = payload.get("autoencoder_contract")
        if contract is None:
            extra = payload.get("extra")
            if isinstance(extra, dict):
                contract = extra.get("autoencoder_contract")
    if not isinstance(contract, dict):
        return None

    if "scaling_factor" in contract:
        vae.scaling_factor = float(contract["scaling_factor"])

    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    current = extract_autoencoder_contract(vae, cfg, input_normalize=input_normalize)
    for key in ("input_normalize", "latent_norm", "data_range"):
        expected = training_cfg.get(key) if key in {"input_normalize", "latent_norm"} and isinstance(training_cfg, dict) else current.get(key)
        actual = contract.get(key)
        if expected is None or actual is None:
            continue
        if str(expected) != str(actual):
            raise ValueError(
                f"Autoencoder checkpoint contract mismatch for {key!r}: config/runtime expects {expected!r}, "
                f"checkpoint recorded {actual!r}."
            )
    return contract


def sync_autoencoder_input_range(
    vae: BaseAutoencoder,
    cfg: dict | None = None,
    *,
    input_normalize: str | None = None,
) -> str:
    """Synchronize model `input_range` with trainer-side `input_normalize`.

    If `model.input_range` is explicitly set in config, preserve it and warn on
    contradiction. If it is omitted, derive a compatible range from
    `training.input_normalize` when possible.
    """
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    explicit_input_range = None
    if isinstance(model_cfg, dict) and "input_range" in model_cfg:
        explicit_input_range = model_cfg.get("input_range")
    normalize_mode = input_normalize
    if normalize_mode is None and isinstance(training_cfg, dict):
        normalize_mode = training_cfg.get("input_normalize")
    derived_range = resolve_model_input_range_from_normalize(normalize_mode)

    current_range = str(getattr(vae, "input_range", "minus_one_to_one"))
    if explicit_input_range is not None:
        current_range = str(explicit_input_range)
        vae.input_range = current_range
        if derived_range is not None and str(current_range).lower() != str(derived_range).lower():
            logging.warning(
                "VAE config sets model.input_range=%r but training.input_normalize=%r implies %r. "
                "Preserving explicit model.input_range.",
                current_range,
                normalize_mode,
                derived_range,
            )
        return current_range

    if derived_range is not None:
        vae.input_range = derived_range
        return derived_range

    vae.input_range = current_range
    return current_range


def apply_input_normalize(x: torch.Tensor, mode: str = "centered") -> torch.Tensor:
    """Normalize image-space inputs before VAE encoding.

    Args:
        x: Input tensor in canonical image space, typically [0, 1].
        mode: One of {"centered", "positive", "zscore"}.
    """
    normalized = str(mode).lower()
    if normalized in {"centered", "symmetric"}:
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
        "Expected one of: 'centered', 'symmetric', 'positive', 'zscore'."
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
    return posterior.mode() * resolve_latent_scaling_factor(vae)


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
    "resolve_model_input_range_from_normalize",
    "resolve_latent_scaling_factor",
    "extract_autoencoder_contract",
    "apply_autoencoder_checkpoint_contract",
    "sync_autoencoder_input_range",
    "apply_input_normalize",
    "encode_to_latent",
    "decode_from_latent",
    "reconstruct_from_image",
]
