from __future__ import annotations

from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, StepLR

from core.registry import Registry


LR_SCHEDULER_REGISTRY = Registry[Callable[..., LRScheduler]]("lr_schedulers")


@LR_SCHEDULER_REGISTRY.register("steplr")
def _build_step_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    return StepLR(
        optimizer,
        step_size=int(params.get("step_size", 10)),
        gamma=float(params.get("gamma", 0.1)),
    )


@LR_SCHEDULER_REGISTRY.register("cosineannealinglr")
def _build_cosine_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    return CosineAnnealingLR(
        optimizer,
        T_max=int(params.get("T_max", params.get("epochs", 100))),
    )


@LR_SCHEDULER_REGISTRY.register("exponentiallr")
def _build_exponential_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    return ExponentialLR(
        optimizer,
        gamma=float(params.get("gamma", 0.99)),
    )


def build_lr_scheduler(optimizer: Optimizer, config: dict) -> LRScheduler | None:
    """Build an LR scheduler from config. Returns None if not configured."""
    spec = config.get("scheduler")
    if spec is None or (isinstance(spec, str) and spec.lower() in {"none", ""}):
        return None

    if isinstance(spec, str):
        name = spec.lower()
        params: dict = {}
    elif isinstance(spec, dict):
        name = str(spec.get("name", "none")).lower()
        params = dict(spec.get("params", {}))
        if name in {"none", ""}:
            return None
    else:
        raise TypeError(
            "training.scheduler must be either a string, a dict, or omitted/null."
        )

    factory = LR_SCHEDULER_REGISTRY.get(name)
    return factory(optimizer, params)
