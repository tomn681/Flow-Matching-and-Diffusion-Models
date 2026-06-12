from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("x0_denoising")
class X0DenoisingNoise:
    """Plain x0-regression denoising.

    This is not consistency training. The model learns to reconstruct the clean
    sample directly from a noised input, and therefore requires
    `scheduler.config.prediction_type == "sample"`.
    """

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        validate_noise_scheduler_contract("x0_denoising", scheduler)

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


NOISE_REGISTRY.register_value("consistency", X0DenoisingNoise)
