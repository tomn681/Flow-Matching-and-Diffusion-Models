from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch


ConditioningAdapter = Callable[[torch.Tensor, torch.Tensor | None, str | None], tuple[torch.Tensor, torch.Tensor | None]]


@dataclass(frozen=True)
class ChainAdapterSpec:
    key: str
    adapter: ConditioningAdapter


class ConditioningChain:
    """Compose conditioning adapters and merge all cross-attention contexts."""

    def __init__(self, adapters: list[ChainAdapterSpec]) -> None:
        self.adapters = list(adapters)

    def __call__(
        self,
        model_input: torch.Tensor,
        conditioning: Mapping[str, torch.Tensor | None] | torch.Tensor | None,
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
        conditioning: Mapping[str, torch.Tensor | None] | torch.Tensor | None,
        key: str,
    ) -> torch.Tensor | None:
        if conditioning is None:
            return None
        if isinstance(conditioning, Mapping):
            return conditioning.get(key)
        return conditioning


__all__ = ["ChainAdapterSpec", "ConditioningChain"]
