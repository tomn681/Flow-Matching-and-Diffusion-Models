from __future__ import annotations

import torch

from core.types import NoisyBatch
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("rectified_flow")
class RectifiedFlowNoise:
    """Rectified-flow style interpolation from noise to data with velocity target."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        t = torch.rand(clean.size(0), device=device)
        t_view = t.view(clean.size(0), *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * noise + t_view * clean
        target = clean - noise
        timesteps = (t * (self.scheduler.config.num_train_timesteps - 1)).long()
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)
