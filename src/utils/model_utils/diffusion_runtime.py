"""
Runtime encode/decode helpers for diffusion/flow-like samplers and trainers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from core import NoisingScheduler
from core.noise_contracts import effective_noise_family_for_config
from pipelines.utils import build_scheduler, resolve_conditioning_mode, resolve_scheduler_override, sample_with_scheduler
from utils.utils import select_visual_indices


_RESIDUAL_FLOW_FAMILIES = frozenset({"residual_flow_matching", "residual_rectified_flow", "residual_reflow"})
_TRAIN_RUNTIME_MODES = frozenset({"train", "training"})


def _extract_tensor_conditioning(
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    if torch.is_tensor(conditioning_batch):
        return conditioning_batch
    if isinstance(conditioning_batch, Mapping):
        value = conditioning_batch.get("concatenate")
        if torch.is_tensor(value):
            return value
    return None


def _validate_reference_shape(reference: torch.Tensor, batch_shape: tuple[int, ...], label: str) -> None:
    if tuple(reference.shape) != tuple(batch_shape):
        raise ValueError(
            f"{label} shape {tuple(reference.shape)} does not match requested batch_shape {tuple(batch_shape)}."
        )


def _resolve_reference_batch(
    *,
    runtime_mode: str,
    batch_shape: tuple[int, ...],
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None,
    target_batch: torch.Tensor | None,
    reference_batch: torch.Tensor | None,
) -> torch.Tensor:
    mode = str(runtime_mode or "inference").strip().lower()
    if mode in _TRAIN_RUNTIME_MODES:
        reference = reference_batch if reference_batch is not None else target_batch
        if reference is None:
            raise ValueError("Training-mode reference initialization requires target_batch or reference_batch.")
        _validate_reference_shape(reference, batch_shape, "Training reference batch")
        return reference

    reference = _extract_tensor_conditioning(conditioning_batch)
    if reference is None:
        raise ValueError(
            "Inference reference initialization requires tensor-valued conditioning; "
            "target_batch is not used as an inference source."
        )
    _validate_reference_shape(reference, batch_shape, "Inference conditioning batch")
    return reference


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
    cfg_rescale: float = 0.0,
    target_batch: torch.Tensor | None = None,
    reference_batch: torch.Tensor | None = None,
    init_from_reference: bool = False,
    init_image_batch: torch.Tensor | None = None,
    strength: float = 1.0,
    scheduler_override: str | None = None,
    runtime_mode: str = "inference",
) -> torch.Tensor:
    noise_family = effective_noise_family_for_config(
        str(model_cfg.get("model_type", "")),
        training_cfg=training_cfg,
        model_cfg=model_cfg,
    )
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
        noise_family=noise_family,
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
    if init_from_reference:
        if selected_timesteps.numel() == 0:
            raise ValueError("No timesteps selected after applying start_step/last_n_steps.")
        reference = _resolve_reference_batch(
            runtime_mode=runtime_mode,
            batch_shape=batch_shape,
            conditioning_batch=conditioning_batch,
            target_batch=target_batch,
            reference_batch=reference_batch,
        )
        if isinstance(scheduler, NoisingScheduler):
            t0 = selected_timesteps[0]
            timesteps = t0.expand(reference.size(0)).to(reference.device)
            noise = torch.randn_like(reference)
            init_sample = scheduler.add_noise(reference, noise, timesteps).to(device)
        else:
            logging.warning(
                "Requested init_from_reference but scheduler '%s' has no add_noise; falling back to random init.",
                scheduler.__class__.__name__,
            )
    if noise_family in _RESIDUAL_FLOW_FAMILIES:
        if not torch.is_tensor(conditioning_batch):
            raise ValueError(
                f"{noise_family} sampling requires tensor-valued conditioning_batch to use as the source endpoint."
            )
        init_sample = conditioning_batch.to(device)
        conditioning_batch = None
        init_image_batch = None
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )
    if noise_family in _RESIDUAL_FLOW_FAMILIES:
        conditioning_mode = None
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
        cfg_rescale=cfg_rescale,
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
