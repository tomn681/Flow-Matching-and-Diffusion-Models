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
        start_epoch: int = 0,
        data_range: str = "zero_to_one",
    ) -> None:
        super().__init__(weight=weight)
        self.start_epoch = int(start_epoch)
        self.loss = PerceptualLoss(
            resize=resize,
            backbone=backbone,
            use_lpips=use_lpips,
            lpips_net=lpips_net,
            data_range=data_range,
        )
        self._device = torch.device("cpu")

    def is_active(self, epoch: int, global_step: int) -> bool:
        del global_step
        return epoch >= self.start_epoch

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
