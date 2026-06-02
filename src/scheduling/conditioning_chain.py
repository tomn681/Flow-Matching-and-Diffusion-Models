from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch


ConditioningAdapter = Callable[[torch.Tensor, object | None, str | None], tuple[torch.Tensor, torch.Tensor | None]]


@dataclass(frozen=True)
class ChainAdapterSpec:
    key: str
    adapter: ConditioningAdapter


class ConditioningChain:
    """Compose conditioning adapters and merge all cross-attention contexts.

    Example:
        chain = ConditioningChain(
            [
                ChainAdapterSpec("concatenate", concat_adapter),
                ChainAdapterSpec("attention", attention_adapter),
            ]
        )
        model_input, context = chain(
            noisy_latents,
            {"concatenate": cond_image, "attention": text_embeddings},
            latent_norm="standardize",
        )
    """

    def __init__(self, adapters: list[ChainAdapterSpec]) -> None:
        self.adapters = list(adapters)

    def __call__(
        self,
        model_input: torch.Tensor,
        conditioning: Mapping[str, object | None] | object | None,
        latent_norm: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        context_parts: list[torch.Tensor] = []
        for spec in self.adapters:
            cond = self._resolve_conditioning_for_adapter(conditioning, spec.key)
            model_input, context = spec.adapter(model_input, cond, latent_norm)
            if context is not None:
                context_parts.append(context)
        combined_context = torch.cat(context_parts, dim=1) if context_parts else None
        return model_input, combined_context

    @staticmethod
    def _resolve_conditioning_for_adapter(
        conditioning: Mapping[str, object | None] | object | None,
        key: str,
    ) -> object | None:
        if conditioning is None:
            return None
        if isinstance(conditioning, Mapping):
            return conditioning.get(key)
        return conditioning


__all__ = ["ChainAdapterSpec", "ConditioningChain"]
