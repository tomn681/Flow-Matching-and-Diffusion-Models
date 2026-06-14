from __future__ import annotations

import math
import warnings
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR, LRScheduler, StepLR

from core.registry import Registry


LR_SCHEDULER_REGISTRY = Registry[Callable[..., LRScheduler]]("lr_schedulers")


@LR_SCHEDULER_REGISTRY.register("steplr")
def _build_step_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    step_size = int(params.get("step_size", 10))
    horizon = int(params.get("epochs", 1))
    if step_size < max(1, horizon):
        warnings.warn(
            f"StepLR step_size={step_size} is shorter than training horizon epochs={horizon}.",
            stacklevel=2,
        )
    return StepLR(
        optimizer,
        step_size=step_size,
        gamma=float(params.get("gamma", 0.1)),
    )


@LR_SCHEDULER_REGISTRY.register("cosineannealinglr")
def _build_cosine_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    t_max = int(params.get("T_max", params.get("epochs", 100)))
    horizon = int(params.get("epochs", 100))
    if t_max < max(1, horizon):
        warnings.warn(
            f"CosineAnnealingLR T_max={t_max} is shorter than training horizon epochs={horizon}. "
            "The schedule will cycle or flatten before training ends.",
            stacklevel=2,
        )
    return CosineAnnealingLR(
        optimizer,
        T_max=t_max,
    )


@LR_SCHEDULER_REGISTRY.register("exponentiallr")
def _build_exponential_lr(optimizer: Optimizer, params: dict) -> LRScheduler:
    return ExponentialLR(
        optimizer,
        gamma=float(params.get("gamma", 0.99)),
    )


def _resolve_total_steps(params: dict) -> int:
    total_steps = params.get("total_steps")
    if total_steps is not None:
        return max(1, int(total_steps))
    steps_per_epoch = int(params.get("steps_per_epoch", 0))
    epochs = int(params.get("epochs", 1))
    return max(1, steps_per_epoch * epochs) if steps_per_epoch > 0 else max(1, epochs)


def _resolve_warmup_steps(params: dict) -> int:
    return max(0, int(params.get("warmup_steps", params.get("lr_warmup_steps", 0))))


def _tag_step_scheduler(scheduler: LRScheduler) -> LRScheduler:
    setattr(scheduler, "_step_unit", "step")
    return scheduler


@LR_SCHEDULER_REGISTRY.register("warmup_linear")
def _build_warmup_linear(optimizer: Optimizer, params: dict) -> LRScheduler:
    total_steps = _resolve_total_steps(params)
    warmup_steps = min(_resolve_warmup_steps(params), max(0, total_steps - 1))

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        remaining = max(0, total_steps - current_step)
        return float(remaining) / float(decay_steps)

    return _tag_step_scheduler(LambdaLR(optimizer, _lr_lambda))


@LR_SCHEDULER_REGISTRY.register("warmup_cosine")
def _build_warmup_cosine(optimizer: Optimizer, params: dict) -> LRScheduler:
    total_steps = _resolve_total_steps(params)
    warmup_steps = min(_resolve_warmup_steps(params), max(0, total_steps - 1))

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _tag_step_scheduler(LambdaLR(optimizer, _lr_lambda))


def build_lr_scheduler(optimizer: Optimizer, config: dict) -> LRScheduler | None:
    """Build an LR scheduler from config. Returns None if not configured."""
    spec = config.get("lr_scheduler")
    if spec is None and "scheduler" in config:
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
            "training.lr_scheduler must be either a string, a dict, or omitted/null."
        )

    merged_params = {
        "epochs": config.get("epochs", 1),
        "steps_per_epoch": config.get("steps_per_epoch", 0),
        "lr_warmup_steps": config.get("lr_warmup_steps", 0),
    }
    merged_params.update(params)

    factory = LR_SCHEDULER_REGISTRY.get(name)
    return factory(optimizer, merged_params)
