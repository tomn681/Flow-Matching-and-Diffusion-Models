from __future__ import annotations

from typing import Callable

import torch

from core.registry import Registry
from .sampling_loop import _prepare_attention_context, normalize_latent_conditioning


ConditioningAdapter = Callable[[torch.Tensor, torch.Tensor | None, str | None], tuple[torch.Tensor, torch.Tensor | None]]

CONDITIONING_ADAPTER_REGISTRY = Registry[ConditioningAdapter]("conditioning_adapters")


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


def resolve_conditioning_adapter(mode: str | None) -> ConditioningAdapter:
    key = str(mode or "none").strip().lower() or "none"
    return CONDITIONING_ADAPTER_REGISTRY.get(key)

