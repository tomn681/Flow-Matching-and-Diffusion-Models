from __future__ import annotations

import torch

from core.types import NoisyBatch
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("edm")
class EDMNoise:
    """EDM-style sigma perturbation: noisy = clean + sigma * noise, target = noise."""

    def __init__(
        self,
        scheduler,
        *,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> None:
        self.scheduler = scheduler
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

    def _sample_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device)
        inv_rho = 1.0 / self.rho
        smin = self.sigma_min ** inv_rho
        smax = self.sigma_max ** inv_rho
        return (smax + u * (smin - smax)) ** self.rho

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        sigmas = self._sample_sigmas(clean.size(0), device)
        sigma_view = sigmas.view(clean.size(0), *([1] * (clean.dim() - 1)))
        noisy = clean + sigma_view * noise
        max_steps = max(1, int(self.scheduler.config.num_train_timesteps) - 1)
        sigma_min = max(self.sigma_min, 1e-12)
        sigma_max = max(self.sigma_max, sigma_min + 1e-12)
        normalized = (torch.log(sigmas) - torch.log(torch.tensor(sigma_min, device=device))) / (
            torch.log(torch.tensor(sigma_max, device=device)) - torch.log(torch.tensor(sigma_min, device=device))
        )
        timesteps = torch.clamp((normalized * max_steps).long(), 0, max_steps)
        return NoisyBatch(noisy=noisy, target=noise, timesteps=timesteps)
