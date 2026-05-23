from __future__ import annotations

import torch

from core.types import NoisyBatch
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("consistency")
class ConsistencyNoise:
    """Consistency-style corruption: model learns to reconstruct clean from noisy samples."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (clean.size(0),),
            device=device,
        ).long()
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return NoisyBatch(noisy=noisy, target=clean, timesteps=timesteps)
