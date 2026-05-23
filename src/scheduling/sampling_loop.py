from __future__ import annotations

import math
import time
from typing import Tuple

import torch


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


def normalize_latent_conditioning(
    condition: torch.Tensor | None, mode: str | None
) -> torch.Tensor | None:
    """Apply per-sample normalization for latent conditioning."""
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
    guidance_scale: float = 1.0,
    unconditional_conditioning_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a generative sampling loop using the provided scheduler and model."""
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    if start_step is not None:
        start_step = int(start_step)
        if start_step < 0:
            raise ValueError("start_step must be >= 0.")
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
    uncond = _align_conditioning(unconditional_conditioning_batch, current.size(0))
    if conditioning_mode == "attention":
        cond = normalize_latent_conditioning(cond, latent_norm)
        if uncond is not None:
            uncond = normalize_latent_conditioning(uncond, latent_norm)
    attention_ctx = _prepare_attention_context(cond) if conditioning_mode == "attention" else None
    uncond_attention_ctx = _prepare_attention_context(uncond) if conditioning_mode == "attention" and uncond is not None else None

    for t in timesteps:
        model_input = current
        if conditioning_mode == "concatenate" and cond is not None:
            model_input = torch.cat([model_input, cond], dim=1)
        step_t = t if torch.is_tensor(t) else torch.as_tensor(t, device=current.device)
        if torch.is_tensor(step_t) and step_t.device != current.device:
            step_t = step_t.to(current.device)
        if step_t.dim() == 0:
            step_t = step_t.expand(current.size(0))

        sync_if_cuda(current.device)
        start = time.perf_counter()
        if guidance_scale != 1.0 and conditioning_mode in {"attention", "concatenate"} and cond is not None:
            if conditioning_mode == "concatenate":
                uncond_cat = uncond if uncond is not None else torch.zeros_like(cond)
                cond_input = torch.cat([current, cond], dim=1)
                uncond_input = torch.cat([current, uncond_cat], dim=1)
                pred_cond = _forward_model(model, cond_input, step_t, context_ca=None)
                pred_uncond = _forward_model(model, uncond_input, step_t, context_ca=None)
            else:
                pred_cond = _forward_model(model, model_input, step_t, context_ca=attention_ctx)
                pred_uncond = _forward_model(
                    model,
                    current,
                    step_t,
                    context_ca=uncond_attention_ctx,
                )
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred = _forward_model(model, model_input, step_t, context_ca=attention_ctx)
        sync_if_cuda(current.device)

        if timing is not None:
            timing["model_seconds"] = timing.get("model_seconds", 0.0) + (time.perf_counter() - start)
            timing["model_calls"] = timing.get("model_calls", 0) + 1

        step = scheduler.step(pred, t, current)
        current = step.prev_sample

    return current


__all__ = [
    "sync_if_cuda",
    "normalize_latent_conditioning",
    "_prepare_attention_context",
    "sample_with_scheduler",
]
