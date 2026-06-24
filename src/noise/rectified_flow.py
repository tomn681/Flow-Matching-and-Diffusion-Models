from __future__ import annotations

import torch

from core.types import NoisyBatch
from core.noise_contracts import validate_noise_scheduler_contract
from .base import BaseNoiseProcess
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("rectified_flow")
class RectifiedFlowNoise(BaseNoiseProcess):
    """Rectified-flow alias over the canonical flow-matching convention.

    This implementation uses the same timestep/noise contract as `flow_matching`:
    `t` is the noise fraction, `noisy=(1-t)*clean + t*noise`, and the model
    predicts `noise-clean`. The distinction between flow matching and rectified
    flow only exists once the coupling differs; for independent coupling the
    objectives are the same.
    """

    def __init__(
        self,
        scheduler,
        *,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        shift: float | None = None,
    ) -> None:
        super().__init__(scheduler)
        validate_noise_scheduler_contract("rectified_flow", scheduler)
        self.timestep_sampling = str(timestep_sampling).strip().lower()
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.shift = float(
            shift
            if shift is not None
            else getattr(getattr(scheduler, "config", None), "shift", 1.0) or 1.0
        )

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.timestep_sampling in {"logit_normal", "lognormal_logit"}:
            normal = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(normal)
        else:
            raise ValueError(
                f"Unsupported rectified-flow timestep_sampling '{self.timestep_sampling}'. "
                "Expected one of {'uniform', 'logit_normal'}."
            )
        if self.shift > 0.0 and self.shift != 1.0:
            t = (self.shift * t) / (1.0 + (self.shift - 1.0) * t)
        return t.clamp(1e-5, 1.0 - 1e-5)

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean)
        t = self._sample_t(clean.size(0), device)
        t_view = t.view(clean.size(0), *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * clean + t_view * noise
        target = noise - clean
        timesteps = t * float(self.scheduler.config.num_train_timesteps - 1)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)
