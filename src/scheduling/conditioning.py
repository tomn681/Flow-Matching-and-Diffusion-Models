from __future__ import annotations

from typing import Callable

import torch
from models.vae.constants import LATENT_SCALE

from core.registry import Registry
from .sampling_loop import _prepare_attention_context, normalize_latent_conditioning


ConditioningAdapter = Callable[[torch.Tensor, torch.Tensor | None, str | None], tuple[torch.Tensor, torch.Tensor | None]]

CONDITIONING_ADAPTER_REGISTRY = Registry[ConditioningAdapter]("conditioning_adapters")


class LatentAttentionAdapter:
    """Attention adapter that can encode conditioning images through a frozen VAE."""

    def __init__(self) -> None:
        self._vae_model = None

    def set_vae_model(self, vae_model) -> None:
        self._vae_model = vae_model

    def clear_vae_model(self) -> None:
        self._vae_model = None

    def _encode_conditioning(self, cond: torch.Tensor) -> torch.Tensor:
        if self._vae_model is None:
            return cond
        model_in = self._vae_model.image_to_model_range(cond) if hasattr(self._vae_model, "image_to_model_range") else cond
        try:
            encoded = self._vae_model.encode(model_in, normalize=True)
            if isinstance(encoded, torch.Tensor):
                return encoded
        except TypeError:
            pass
        posterior = self._vae_model.encode(model_in, normalize=False)
        if isinstance(posterior, torch.Tensor):
            return posterior
        if not hasattr(posterior, "mode"):
            raise TypeError(f"Unsupported VAE encode output type '{type(posterior).__name__}'.")
        return posterior.mode() * LATENT_SCALE

    def __call__(
        self,
        model_input: torch.Tensor,
        cond: torch.Tensor | None,
        latent_norm: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cond is None:
            return model_input, None
        encoded = self._encode_conditioning(cond)
        normalized = normalize_latent_conditioning(encoded, latent_norm)
        context = _prepare_attention_context(normalized)
        return model_input, context


@CONDITIONING_ADAPTER_REGISTRY.register("none")
def _adapt_none(model_input: torch.Tensor, cond: torch.Tensor | None, latent_norm: str | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model_input, None


@CONDITIONING_ADAPTER_REGISTRY.register("concatenate")
def _adapt_concatenate(
    model_input: torch.Tensor, cond: torch.Tensor | None, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is None:
        return model_input, None
    return torch.cat([model_input, cond], dim=1), None


@CONDITIONING_ADAPTER_REGISTRY.register("attention")
def _adapt_attention(
    model_input: torch.Tensor, cond: torch.Tensor | None, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is None:
        return model_input, None
    normalized = normalize_latent_conditioning(cond, latent_norm)
    context = _prepare_attention_context(normalized)
    return model_input, context


_LATENT_ATTENTION_ADAPTER = LatentAttentionAdapter()
CONDITIONING_ADAPTER_REGISTRY.register("latent_attention")(_LATENT_ATTENTION_ADAPTER)


def configure_latent_attention_vae(vae_model) -> None:
    _LATENT_ATTENTION_ADAPTER.set_vae_model(vae_model)


def clear_latent_attention_vae() -> None:
    _LATENT_ATTENTION_ADAPTER.clear_vae_model()


def resolve_conditioning_adapter(mode: str | None) -> ConditioningAdapter:
    key = str(mode or "none").strip().lower() or "none"
    return CONDITIONING_ADAPTER_REGISTRY.get(key)
