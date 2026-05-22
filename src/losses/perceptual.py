from __future__ import annotations

import torch

from nn.losses.perceptual import PerceptualLoss
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("perceptual")
class PerceptualLossComponent(BaseLossComponent):
    name = "perceptual"

    def __init__(self, weight: float = 1.0, *, resize: bool = True) -> None:
        super().__init__(weight=weight)
        self.loss = PerceptualLoss(resize=resize)

    def to(self, device: torch.device) -> "PerceptualLossComponent":
        self.loss = self.loss.to(device)
        return self

    def compute(self, prediction: torch.Tensor, target: torch.Tensor, **context) -> torch.Tensor:
        return self.loss(prediction, target)
