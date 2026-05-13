"""
Shared helpers for training/sampling pipelines (schedulers, conditioning, checkpoint helpers).
"""

from __future__ import annotations

import math
import time
import inspect
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
    # Keep compatibility across scheduler classes with different ctor kwargs.
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered_params = {k: v for k, v in params.items() if k in allowed}
    scheduler = cls(num_train_timesteps=num_train_steps, **filtered_params)
    num_inference = int(scheduler_cfg.get("num_inference_steps") or training_cfg.get("num_inference_steps") or num_train_steps)
    return scheduler, num_inference


def resolve_scheduler_override(name: str | None) -> Dict | None:
    """
    Map user-facing scheduler aliases into scheduler config overrides.
    """
    if not name:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    alias = {
        "ddpm": {"name": "ddpm"},
        "ddim": {"name": "ddim"},
        "dpmsolver1": {"name": "dpm_multistep", "params": {"solver_order": 1, "algorithm_type": "dpmsolver"}},
        "dpmsolver2": {"name": "dpm_multistep", "params": {"solver_order": 2, "algorithm_type": "dpmsolver"}},
        "dpmsolver++": {"name": "dpm_multistep", "params": {"solver_order": 2, "algorithm_type": "dpmsolver++"}},
        "dpmsolversde": {"name": "dpm_sde"},
        "unipc": {"name": "unipc"},
        "flowmatch": {"name": "flow_match_euler"},
        "flow_match_euler": {"name": "flow_match_euler"},
    }
    if key in alias:
        return alias[key]
    if key in SCHEDULER_REGISTRY:
        return {"name": key}
    available = ", ".join(sorted(list(alias.keys())))
    raise ValueError(f"Unknown scheduler override '{name}'. Available: {available}")


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


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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
    timing: dict | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    init_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run a generative sampling loop using the provided scheduler and model.
    """
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    if start_step is not None:
        start_step = int(start_step)
        if start_step < 0:
            raise ValueError("start_step must be >= 0.")
        # Keep only the denoising tail from the requested training timestep down to 0.
        timesteps = timesteps[timesteps <= start_step]
    if last_n_steps is not None:
        last_n_steps = int(last_n_steps)
        if last_n_steps <= 0:
            raise ValueError("last_n_steps must be > 0.")
        timesteps = timesteps[-last_n_steps:]
    if timesteps.numel() == 0:
        raise ValueError("No timesteps selected after applying start_step/last_n_steps.")

    current = init_sample.to(device) if init_sample is not None else torch.randn(sample_shape, device=device)
    cond = _align_conditioning(conditioning_batch, current.size(0))
    if conditioning_mode == "attention":
        cond = normalize_latent_conditioning(cond, latent_norm)
    attention_ctx = _prepare_attention_context(cond) if conditioning_mode == "attention" else None

    for t in timesteps:
        model_input = current
        if conditioning_mode == "concatenate" and cond is not None:
            model_input = torch.cat([model_input, cond], dim=1)
        timesteps = t if torch.is_tensor(t) else torch.as_tensor(t, device=current.device)
        if torch.is_tensor(timesteps) and timesteps.device != current.device:
            timesteps = timesteps.to(current.device)
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(current.size(0))
        sync_if_cuda(current.device)
        start = time.perf_counter()
        pred = _forward_model(model, model_input, timesteps, context_ca=attention_ctx)
        sync_if_cuda(current.device)
        if timing is not None:
            timing["model_seconds"] = timing.get("model_seconds", 0.0) + (time.perf_counter() - start)
            timing["model_calls"] = timing.get("model_calls", 0) + 1
        step = scheduler.step(pred, t, current)
        current = step.prev_sample
    return current
