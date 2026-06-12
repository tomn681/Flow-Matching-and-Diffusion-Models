"""
Runtime encode/decode helpers for diffusion/flow-like samplers and trainers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from core import NoisingScheduler
from core.noise_contracts import noise_family_for_model_type
from pipelines.utils import build_scheduler, resolve_conditioning_mode, resolve_scheduler_override, sample_with_scheduler
from utils.utils import select_visual_indices


def encode_diffusion_batch(scheduler, targets: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(targets)
    return scheduler.add_noise(targets, noise, timesteps)


def decode_diffusion_batch(
    model,
    training_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    batch_shape: tuple[int, ...],
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    timing: dict | None = None,
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    reference_batch: torch.Tensor | None = None,
    init_from_reference: bool = False,
    init_image_batch: torch.Tensor | None = None,
    strength: float = 1.0,
    scheduler_override: str | None = None,
) -> torch.Tensor:
    scheduler_cfg = dict(model_cfg.get("scheduler", {}))
    override_cfg = resolve_scheduler_override(scheduler_override)
    if override_cfg is not None:
        scheduler_cfg["name"] = override_cfg["name"]
        override_params = dict(override_cfg.get("params", {}))
        merged_params = dict(scheduler_cfg.get("params", {}))
        merged_params.update(override_params)
        scheduler_cfg["params"] = merged_params
    scheduler, num_inference = build_scheduler(
        scheduler_cfg,
        training_cfg,
        noise_family=noise_family_for_model_type(str(model_cfg.get("model_type", ""))),
    )
    if num_inference_steps is not None:
        num_inference = int(num_inference_steps)
    scheduler.set_timesteps(num_inference)
    selected_timesteps = scheduler.timesteps
    if start_step is not None:
        selected_timesteps = selected_timesteps[selected_timesteps <= int(start_step)]
    if last_n_steps is not None:
        selected_timesteps = selected_timesteps[-int(last_n_steps):]

    init_sample = None
    if init_from_reference and reference_batch is not None:
        if selected_timesteps.numel() == 0:
            raise ValueError("No timesteps selected after applying start_step/last_n_steps.")
        if isinstance(scheduler, NoisingScheduler):
            t0 = selected_timesteps[0]
            timesteps = t0.expand(reference_batch.size(0)).to(reference_batch.device)
            noise = torch.randn_like(reference_batch)
            init_sample = scheduler.add_noise(reference_batch, noise, timesteps).to(device)
        else:
            logging.warning(
                "Requested init_from_reference but scheduler '%s' has no add_noise; falling back to random init.",
                scheduler.__class__.__name__,
            )
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )
    latent_norm = training_cfg.get("latent_norm")
    return sample_with_scheduler(
        model,
        scheduler,
        num_inference,
        batch_shape,
        device,
        conditioning_mode=conditioning_mode,
        conditioning_batch=conditioning_batch,
        latent_norm=latent_norm,
        timing=timing,
        start_step=start_step,
        last_n_steps=last_n_steps,
        init_sample=init_sample,
        init_image=init_image_batch,
        strength=strength,
    )


def prepare_diffusion_visual_batch(dataset, count: int, device: torch.device, seed: int | None = None):
    indices = select_visual_indices(dataset, count, seed=seed)
    targets = []
    conditioning = []
    for idx in indices:
        sample = dataset[idx]
        targets.append(sample["target"])
        conditioning.append(sample.get("image"))
    target_batch = torch.stack(targets, dim=0).to(device)
    if conditioning and all(c is not None for c in conditioning):
        cond_batch = torch.stack(conditioning, dim=0).to(device)
    else:
        cond_batch = None
    return target_batch, cond_batch
