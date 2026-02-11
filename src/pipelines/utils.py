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


def _forward_model(model, inputs, timesteps, context_ca=None):
    if context_ca is not None:
        outputs = model(inputs, timesteps, context_ca=context_ca)
    else:
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


def normalize_latent_conditioning(condition: torch.Tensor | None, mode: str | None) -> torch.Tensor | None:
    """
    normalize_latent_conditioning Function

    Applies per-sample normalization for latent conditioning.

    Inputs:
        - condition: (Tensor | None) Conditioning tensor (B, C, H, W).
        - mode: (String | None) One of "standardize", "minmax", or None/"none".

    Outputs:
        - condition: (Tensor | None) Normalized conditioning tensor.
    """
    if condition is None:
        return None
    mode_value = str(mode or "none").lower()
    if mode_value in {"none", "false", "off"}:
        return condition
    eps = 1e-6
    spatial_dims = tuple(range(2, condition.dim()))
    if mode_value == "standardize":
        mean = condition.mean(dim=spatial_dims, keepdim=True)
        std = condition.std(dim=spatial_dims, keepdim=True)
        return (condition - mean) / (std + eps)
    if mode_value == "minmax":
        minv = condition.amin(dim=spatial_dims, keepdim=True)
        maxv = condition.amax(dim=spatial_dims, keepdim=True)
        return (condition - minv) / (maxv - minv + eps)
    raise ValueError(f"Unknown latent_norm mode: {mode}")


def _prepare_attention_context(condition: torch.Tensor | None) -> torch.Tensor | None:
    if condition is None:
        return None
    if condition.dim() == 3:
        return condition
    if condition.dim() >= 4:
        return condition
    raise ValueError(f"Unsupported conditioning shape for attention: {tuple(condition.shape)}")


def sample_with_scheduler(
    model: torch.nn.Module,
    scheduler,
    num_inference_steps: int,
    sample_shape: Tuple[int, ...],
    device: torch.device,
    conditioning_mode: str | None = None,
    conditioning_batch: torch.Tensor | None = None,
    latent_norm: str | None = None,
) -> torch.Tensor:
    """
    Run a generative sampling loop using the provided scheduler and model.
    """
    scheduler.set_timesteps(num_inference_steps)
    current = torch.randn(sample_shape, device=device)
    cond = _align_conditioning(conditioning_batch, current.size(0))
    if conditioning_mode == "attention":
        cond = normalize_latent_conditioning(cond, latent_norm)
    attention_ctx = _prepare_attention_context(cond) if conditioning_mode == "attention" else None

    for t in scheduler.timesteps:
        model_input = current
        if conditioning_mode == "concatenate" and cond is not None:
            model_input = torch.cat([model_input, cond], dim=1)
        pred = _forward_model(model, model_input, t, context_ca=attention_ctx)
        step = scheduler.step(pred, t, current)
        current = step.prev_sample
    return current
