from __future__ import annotations

import torch

from nn.losses.ssim import ssim_loss
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ssim")
class SSIMLoss(BaseLossComponent):
    """Differentiable SSIM loss component over reconstruction images."""

    name = "recon_ssim"

    def __init__(
        self,
        weight: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        *,
        start_epoch: int = 0,
    ) -> None:
        super().__init__(weight=weight)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.start_epoch = int(start_epoch)

    def is_active(self, epoch: int, global_step: int) -> bool:
        del global_step
        return epoch >= self.start_epoch

    def compute(self, *, context: dict) -> torch.Tensor:
        pred = context["reconstruction_image"]
        target = context["target"]
        return ssim_loss(pred, target, window_size=self.window_size, sigma=self.sigma)
