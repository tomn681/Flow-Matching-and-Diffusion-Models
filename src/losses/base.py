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

    def compute(self, *, context: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def is_active(self, epoch: int, global_step: int) -> bool:
        return True


class LossAssembler:
    """Compose multiple loss components into a weighted scalar objective."""

    def __init__(self, components: list[BaseLossComponent]) -> None:
        self.components = components

    def metric_keys(self) -> list[str]:
        """All metric names that could appear in assembled parts, stable order."""
        seen: list[str] = []
        for component in self.components:
            if component.name not in seen:
                seen.append(component.name)
        return seen

    def __call__(
        self,
        *,
        context: dict[str, Any],
        epoch: int = 0,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = context["device"]
        dtype = context.get("dtype", torch.float32)
        total = torch.tensor(0.0, device=device, dtype=dtype)
        parts: dict[str, torch.Tensor] = {}

        for component in self.components:
            if not component.is_active(epoch=epoch, global_step=global_step):
                continue
            raw = component.compute(context=context)
            weighted = raw * component.weight
            if component.name in parts:
                parts[component.name] = parts[component.name] + weighted
            else:
                parts[component.name] = weighted
            total = total + weighted

        return total, parts
