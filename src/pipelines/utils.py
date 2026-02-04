"""
Shared helpers for training/sampling pipelines (schedulers, conditioning, checkpoint helpers).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

SCHEDULER_REGISTRY: Dict[str, type] = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "dpm_sde": DPMSolverSDEScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
    "flowmatch": FlowMatchEulerDiscreteScheduler,
}


def resolve_conditioning_mode(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value if value else None


def build_scheduler(spec: Dict, training_cfg: Dict) -> Tuple[object, int]:
    """
    Instantiate a Diffusers scheduler based on config dictionaries.
    Returns the scheduler instance and the number of inference steps.
    """
    scheduler_cfg = dict(spec or {})
    training_cfg = dict(training_cfg or {})
    name = scheduler_cfg.get("name") or training_cfg.get("scheduler") or "ddpm"
    key = str(name).lower()
    if key not in SCHEDULER_REGISTRY:
        available = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    cls = SCHEDULER_REGISTRY[key]
    num_train_steps = int(scheduler_cfg.get("num_train_timesteps") or training_cfg.get("num_train_timesteps") or 1000)
    params = dict(scheduler_cfg.get("params", {}))
    scheduler = cls(num_train_timesteps=num_train_steps, **params)
    num_inference = int(scheduler_cfg.get("num_inference_steps") or training_cfg.get("num_inference_steps") or num_train_steps)
    return scheduler, num_inference


def collect_conditioning_batch(dataset, count: int, device: torch.device) -> torch.Tensor | None:
    """
    Assemble a tensor batch of LDCT conditioning images from the dataset.
    """
    collected = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample.get("image") is None:
            continue
        collected.append(sample["image"])
        if len(collected) >= count:
            break
    if not collected:
        return None
    batch = torch.stack(collected, dim=0)[:count]
    return batch.to(device)


def _forward_model(model, inputs, timesteps):
    outputs = model(inputs, timesteps)
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "sample"):
        return outputs.sample
    return outputs


def _align_conditioning(condition, target_batch):
    if condition is None:
        return None
    if condition.size(0) == target_batch:
        return condition
    repeats = math.ceil(target_batch / condition.size(0))
    conditioned = condition
    if repeats > 1:
        conditioned = condition.repeat(repeats, 1, 1, 1)
    return conditioned[:target_batch]


def sample_with_scheduler(
    model: torch.nn.Module,
    scheduler,
    num_inference_steps: int,
    sample_shape: Tuple[int, ...],
    device: torch.device,
    conditioning_mode: str | None = None,
    conditioning_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run a generative sampling loop using the provided scheduler and model.
    """
    scheduler.set_timesteps(num_inference_steps)
    current = torch.randn(sample_shape, device=device)
    cond = _align_conditioning(conditioning_batch, current.size(0))

    for t in scheduler.timesteps:
        model_input = current
        if conditioning_mode == "concatenate" and cond is not None:
            model_input = torch.cat([model_input, cond], dim=1)
        pred = _forward_model(model, model_input, t)
        step = scheduler.step(pred, t, current)
        current = step.prev_sample
    return current
