from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from scheduling.edm import karras_sigmas, sigma_to_timestep
from .base import BaseNoiseProcess
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("consistency")
class ConsistencyNoise(BaseNoiseProcess):
    """Consistency-training noise process using adjacent sigma pairs.

    The trainer consumes `sigmas`, `next_sigmas`, and `noisy_next` from the
    returned `NoisyBatch.extra` payload.
    """

    def __init__(
        self,
        scheduler,
        *,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> None:
        super().__init__(scheduler)
        validate_noise_scheduler_contract("consistency", scheduler)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_train_timesteps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))

    def _sigma_grid(self, device: torch.device) -> torch.Tensor:
        return karras_sigmas(
            self.num_train_timesteps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        sigma_grid = self._sigma_grid(device)
        upper = max(1, sigma_grid.numel() - 2)
        indices = torch.randint(0, upper, (clean.size(0),), device=device)
        sigmas = sigma_grid[indices]
        next_sigmas = sigma_grid[indices + 1]
        noise = torch.randn_like(clean)
        sigma_view = sigmas.view(clean.size(0), *([1] * (clean.dim() - 1))).to(dtype=clean.dtype)
        next_sigma_view = next_sigmas.view(clean.size(0), *([1] * (clean.dim() - 1))).to(dtype=clean.dtype)
        noisy = clean + sigma_view * noise
        noisy_next = clean + next_sigma_view * noise
        timesteps = sigma_to_timestep(
            sigmas,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_train_timesteps=self.num_train_timesteps,
        )
        next_timesteps = sigma_to_timestep(
            next_sigmas,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_train_timesteps=self.num_train_timesteps,
        )
        return NoisyBatch(
            noisy=noisy,
            target=clean,
            timesteps=timesteps.to(device=device, dtype=clean.dtype),
            extra={
                "sigmas": sigmas,
                "next_sigmas": next_sigmas,
                "noisy_next": noisy_next,
                "next_timesteps": next_timesteps.to(device=device, dtype=clean.dtype),
                "noise": noise,
            },
        )


@NOISE_REGISTRY.register("x0_denoising")
class X0DenoisingNoise(BaseNoiseProcess):
    """Plain x0-regression denoising.

    This remains available as the simple x0-target baseline. It is not
    consistency training.
    """

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)
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
