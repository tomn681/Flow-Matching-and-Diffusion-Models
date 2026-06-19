from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .families import model_family_for_model_type


def canonical_noise_family(noise_key: str) -> str:
    key = str(noise_key).strip().lower()
    return key


def noise_family_for_model_type(model_type: str | None) -> str | None:
    family = model_family_for_model_type(model_type)
    return None if family is None else family.noise_family


def validate_noise_scheduler_contract(noise_key: str, scheduler) -> None:
    family = canonical_noise_family(noise_key)

    if family == "edm":
        if getattr(getattr(scheduler, "config", None), "num_train_timesteps", None) is None:
            raise ValueError(
                f"Noise family '{noise_key}' requires a scheduler/config with num_train_timesteps for sigma discretization."
            )
        return

    if family in {"flow_matching", "rectified_flow", "reflow"}:
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError(
                f"Noise family '{noise_key}' requires FlowMatchEulerDiscreteScheduler, got {scheduler.__class__.__name__}."
            )
        return

    prediction_type = str(getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon") or "epsilon").lower()

    if family == "consistency":
        if getattr(getattr(scheduler, "config", None), "num_train_timesteps", None) is None:
            raise ValueError(
                f"Noise family '{noise_key}' requires a scheduler/config with num_train_timesteps for sigma discretization."
            )
        return

    if family == "x0_denoising":
        if prediction_type != "sample":
            raise ValueError(
                f"Noise family '{noise_key}' requires scheduler prediction_type='sample', got '{prediction_type}'."
            )
        return

    if family == "ddpm" and prediction_type not in {"epsilon", "sample", "v_prediction"}:
        raise ValueError(
            f"Noise family 'ddpm' supports scheduler prediction_type in {{epsilon, sample, v_prediction}}, got '{prediction_type}'."
        )


def resolve_ddpm_prediction_target(
    scheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    prediction_type = str(getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon") or "epsilon").lower()
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return clean
    if prediction_type == "v_prediction":
        if not hasattr(scheduler, "get_velocity"):
            raise ValueError(
                f"Scheduler {scheduler.__class__.__name__} declares prediction_type='v_prediction' but does not implement get_velocity(...)."
            )
        return scheduler.get_velocity(clean, noise, timesteps)
    raise ValueError(f"Unsupported scheduler prediction_type '{prediction_type}'.")


def warn_if_legacy_family_alias(model_type: str) -> None:
    return None


__all__ = [
    "canonical_noise_family",
    "noise_family_for_model_type",
    "validate_noise_scheduler_contract",
    "resolve_ddpm_prediction_target",
    "warn_if_legacy_family_alias",
]
