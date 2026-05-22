from __future__ import annotations

from typing import Any

import torch

from .registry import LOSS_REGISTRY


class BaseLossComponent:
    """Convenient base class for framework-level loss components."""

    name: str = "unnamed"
    weight: float = 1.0

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor, **context: Any) -> torch.Tensor:
        raise NotImplementedError

    def is_active(self, epoch: int, global_step: int) -> bool:
        return True


class LossAssembler:
    """Compose multiple loss components into a weighted scalar objective."""

    def __init__(self, components: list[BaseLossComponent]) -> None:
        self.components = components

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: int = 0,
        global_step: int = 0,
        **context: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        parts: dict[str, torch.Tensor] = {}

        for component in self.components:
            if not component.is_active(epoch=epoch, global_step=global_step):
                continue
            raw = component.compute(prediction, target, **context)
            weighted = raw * component.weight
            if component.name in parts:
                parts[component.name] = parts[component.name] + weighted
            else:
                parts[component.name] = weighted
            total = total + weighted

        return total, parts
