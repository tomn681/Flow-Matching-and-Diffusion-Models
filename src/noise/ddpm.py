from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import resolve_ddpm_prediction_target, validate_noise_scheduler_contract
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("ddpm")
class DDPMNoise:
    """DDPM noise process: model predicts added Gaussian noise."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        validate_noise_scheduler_contract("ddpm", scheduler)

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (clean.size(0),),
            device=device,
        ).long()
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        target = resolve_ddpm_prediction_target(self.scheduler, clean, noise, timesteps)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)
