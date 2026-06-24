from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from scheduling.edm import sample_log_normal_sigmas, sigma_to_timestep
from .base import BaseNoiseProcess
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("edm")
class EDMNoise(BaseNoiseProcess):
    """Real EDM training noise process.

    Samples log-normal sigmas, perturbs clean samples as `x + sigma * eps`, and
    carries sigma metadata for preconditioned training.
    """

    def __init__(
        self,
        scheduler,
        *,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__(scheduler)
        validate_noise_scheduler_contract("edm", scheduler)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def _sample_sigmas(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return sample_log_normal_sigmas(
            batch_size,
            device,
            p_mean=self.p_mean,
            p_std=self.p_std,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            dtype=torch.float32 if dtype == torch.float16 else dtype,
        )

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        sigmas = self._sample_sigmas(clean.size(0), device, clean.dtype)
        sigma_view = sigmas.view(clean.size(0), *([1] * (clean.dim() - 1))).to(dtype=clean.dtype)
        noisy = clean + sigma_view * noise
        timesteps = sigma_to_timestep(
            sigmas,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_train_timesteps=int(getattr(self.scheduler.config, "num_train_timesteps", 1000)),
        )
        return NoisyBatch(
            noisy=noisy,
            target=clean,
            timesteps=timesteps.to(device=device, dtype=clean.dtype),
            extra={"sigmas": sigmas.to(device=device), "noise": noise},
        )
