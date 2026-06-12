from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("flow_matching")
class FlowMatchingNoise:
    """Flow-matching process: model predicts velocity target noise - clean."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        validate_noise_scheduler_contract("flow_matching", scheduler)

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        t = torch.rand(clean.size(0), device=device)

        # Broadcast t across non-batch dims.
        while t.ndim < clean.ndim:
            t = t.unsqueeze(-1)

        noisy = (1.0 - t) * clean + t * noise
        target = noise - clean
        timesteps = t.view(clean.size(0)) * float(self.scheduler.config.num_train_timesteps - 1)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)
