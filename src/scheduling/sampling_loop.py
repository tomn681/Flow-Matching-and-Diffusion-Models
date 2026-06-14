from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Tuple

import torch

from core import NoisingScheduler
from core.types import ModelOutput, unwrap_model_prediction


def _forward_model(model, inputs, timesteps, context_ca=None):
    if context_ca is not None:
        outputs = model(inputs, timesteps, context_ca=context_ca)
    else:
        outputs = model(inputs, timesteps)
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, ModelOutput):
        return outputs.reconstruction
    return unwrap_model_prediction(outputs)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _align_conditioning(condition, target_batch):
    if condition is None:
        return None
    if isinstance(condition, Mapping):
        return {k: _align_conditioning(v, target_batch) for k, v in condition.items()}
    if condition.size(0) == target_batch:
        return condition
    repeats = math.ceil(target_batch / condition.size(0))
    conditioned = condition
    if repeats > 1:
        repeat_shape = (repeats,) + (1,) * (condition.dim() - 1)
        conditioned = condition.repeat(*repeat_shape)
    return conditioned[:target_batch]


def _resolve_conditioning_adapter(mode: str | None):
    from .conditioning import resolve_conditioning_adapter

    return resolve_conditioning_adapter(mode)


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
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    latent_norm: str | None = None,
    timing: dict | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    init_sample: torch.Tensor | None = None,
    init_image: torch.Tensor | None = None,
    strength: float = 1.0,
    guidance_scale: float = 1.0,
    unconditional_conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
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
    if not (0.0 <= float(strength) <= 1.0):
        raise ValueError("strength must be in [0, 1].")

    if init_image is not None and init_sample is not None:
        raise ValueError("init_image and init_sample are mutually exclusive.")

    if init_image is not None:
        init_image = init_image.to(device)
        if tuple(init_image.shape) != tuple(sample_shape):
            raise ValueError(
                f"init_image shape {tuple(init_image.shape)} does not match sample_shape {tuple(sample_shape)}."
            )
        if strength == 0.0:
            return init_image
        if not isinstance(scheduler, NoisingScheduler):
            raise ValueError("Scheduler does not support add_noise required for init_image img2img mode.")

        start_idx = int(timesteps.numel() * (1.0 - float(strength)))
        start_idx = min(max(start_idx, 0), timesteps.numel() - 1)
        noise = torch.randn_like(init_image)
        t_start = timesteps[start_idx]
        t_start_batch = (
            t_start.expand(init_image.size(0))
            if torch.is_tensor(t_start) and t_start.dim() == 0
            else t_start
        )
        if torch.is_tensor(t_start_batch):
            t_start_batch = t_start_batch.to(init_image.device)
        current = scheduler.add_noise(init_image, noise, t_start_batch)
        timesteps = timesteps[start_idx:]
    else:
        current = init_sample.to(device) if init_sample is not None else torch.randn(sample_shape, device=device)

    cond = _align_conditioning(conditioning_batch, current.size(0))
    uncond = _align_conditioning(unconditional_conditioning_batch, current.size(0))
    conditioning_adapter = _resolve_conditioning_adapter(conditioning_mode)

    for t in timesteps:
        model_input, attention_ctx = conditioning_adapter(current, cond, latent_norm)
        step_t = t if torch.is_tensor(t) else torch.as_tensor(t, device=current.device)
        if torch.is_tensor(step_t) and step_t.device != current.device:
            step_t = step_t.to(current.device)
        if step_t.dim() == 0:
            step_t = step_t.expand(current.size(0))

        start = time.perf_counter()
        if timing is not None:
            sync_if_cuda(current.device)
        if guidance_scale != 1.0 and cond is not None:
            if uncond is None:
                model_input_uncond, attention_ctx_uncond = conditioning_adapter.null_conditioning(
                    current,
                    cond,
                    latent_norm,
                )
            else:
                model_input_uncond, attention_ctx_uncond = conditioning_adapter(
                    current,
                    uncond,
                    latent_norm,
                )
            pred_cond = _forward_model(model, model_input, step_t, context_ca=attention_ctx)
            pred_uncond = _forward_model(
                model,
                model_input_uncond,
                step_t,
                context_ca=attention_ctx_uncond,
            )
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred = _forward_model(model, model_input, step_t, context_ca=attention_ctx)
        if timing is not None:
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
    "_align_conditioning",
    "sample_with_scheduler",
]
