from __future__ import annotations

import torch
import torch.nn.functional as F

from nn.losses.reconstruction import bce_focal_loss, focal_loss
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("l1")
class L1Loss(BaseLossComponent):
    name = "recon_l1"

    def compute(self, *, context: dict) -> torch.Tensor:
        return F.l1_loss(context["reconstruction_image"], context["target"])


@LOSS_REGISTRY.register("mse")
class MSELoss(BaseLossComponent):
    name = "recon_mse"

    def compute(self, *, context: dict) -> torch.Tensor:
        return F.mse_loss(context["reconstruction_image"], context["target"])


@LOSS_REGISTRY.register("bce")
class BCELoss(BaseLossComponent):
    name = "recon_bce"

    def compute(self, *, context: dict) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(context["reconstruction"], context["target"])


@LOSS_REGISTRY.register("focal")
class FocalLoss(BaseLossComponent):
    name = "recon_focal"

    def __init__(self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__(weight=weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def compute(self, *, context: dict) -> torch.Tensor:
        return focal_loss(context["reconstruction"], context["target"], alpha=self.alpha, gamma=self.gamma, reduction="mean")


@LOSS_REGISTRY.register("bce_focal")
class BCEFocalLoss(BaseLossComponent):
    name = "recon_bce_focal"

    def __init__(self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__(weight=weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def compute(self, *, context: dict) -> torch.Tensor:
        return bce_focal_loss(context["reconstruction"], context["target"], alpha=self.alpha, gamma=self.gamma, reduction="mean")
