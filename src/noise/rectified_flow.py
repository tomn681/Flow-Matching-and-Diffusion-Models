from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("rectified_flow")
class RectifiedFlowNoise:
    """Rectified-flow alias over the canonical flow-matching convention.

    This implementation uses the same timestep/noise contract as `flow_matching`:
    `t` is the noise fraction, `noisy=(1-t)*clean + t*noise`, and the model
    predicts `noise-clean`. The distinction between flow matching and rectified
    flow only exists once the coupling differs; for independent coupling the
    objectives are the same.
    """

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        validate_noise_scheduler_contract("rectified_flow", scheduler)

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        t = torch.rand(clean.size(0), device=device)
        t_view = t.view(clean.size(0), *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * clean + t_view * noise
        target = noise - clean
        timesteps = t * float(self.scheduler.config.num_train_timesteps - 1)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)
