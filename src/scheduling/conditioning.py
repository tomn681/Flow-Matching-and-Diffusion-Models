from __future__ import annotations

from typing import Callable

import torch
from models.autoencoder.base import BaseAutoencoder
from models.autoencoder.utils import encode_to_latent

from core.registry import Registry
from .conditioning_chain import ChainAdapterSpec, ConditioningChain
from .sampling_loop import _prepare_attention_context, normalize_latent_conditioning


ConditioningAdapter = Callable[[torch.Tensor, torch.Tensor | None, str | None], tuple[torch.Tensor, torch.Tensor | None]]

CONDITIONING_ADAPTER_REGISTRY = Registry[ConditioningAdapter]("conditioning_adapters")


class LatentAttentionAdapter:
    """Factory for per-trainer latent-attention adapter instances."""

    @staticmethod
    def create(vae_model: BaseAutoencoder) -> ConditioningAdapter:
        def _adapt(
            model_input: torch.Tensor,
            cond: torch.Tensor | None,
            latent_norm: str | None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if cond is None:
                return model_input, None
            encoded = encode_to_latent(vae_model, cond)
            normalized = normalize_latent_conditioning(encoded, latent_norm)
            context = _prepare_attention_context(normalized)
            return model_input, context

        return _adapt


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


@CONDITIONING_ADAPTER_REGISTRY.register("latent_attention")
def _adapt_latent_attention(
    model_input: torch.Tensor, cond: torch.Tensor | None, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Stateless default registration. Latent trainers should inject a per-instance
    # adapter created via LatentAttentionAdapter.create(vae_model).
    return _adapt_attention(model_input, cond, latent_norm)


@CONDITIONING_ADAPTER_REGISTRY.register("chain")
def _adapt_chain(
    model_input: torch.Tensor, cond: torch.Tensor | dict[str, torch.Tensor] | None, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    chain = ConditioningChain(
        [
            ChainAdapterSpec(key="concatenate", adapter=_adapt_concatenate),
            ChainAdapterSpec(key="attention", adapter=_adapt_attention),
        ]
    )
    return chain(model_input, cond, latent_norm)


def resolve_conditioning_adapter(mode: str | None) -> ConditioningAdapter:
    key = str(mode or "none").strip().lower() or "none"
    return CONDITIONING_ADAPTER_REGISTRY.get(key)
