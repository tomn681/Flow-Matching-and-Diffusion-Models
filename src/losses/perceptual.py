from __future__ import annotations

import torch

from nn.losses.perceptual import PerceptualLoss
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("perceptual")
class PerceptualLossComponent(BaseLossComponent):
    name = "perceptual"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        resize: bool = True,
        backbone: str = "vgg16",
        use_lpips: bool = False,
        lpips_net: str = "vgg",
    ) -> None:
        super().__init__(weight=weight)
        self.loss = PerceptualLoss(
            resize=resize,
            backbone=backbone,
            use_lpips=use_lpips,
            lpips_net=lpips_net,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "PerceptualLossComponent":
        self._device = device
        self.loss = self.loss.to(device)
        return self

    def compute(self, *, context: dict) -> torch.Tensor:
        prediction = context["reconstruction_image"]
        target = context["target"]
        prediction = prediction.to(self._device)
        target = target.to(self._device)
        result = self.loss(prediction, target)
        return result.to(context["device"])
