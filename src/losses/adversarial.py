from __future__ import annotations

import torch

from nn.losses.adversarial import discriminator_hinge_loss, generator_hinge_loss
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("gan_generator")
class GANGeneratorLoss(BaseLossComponent):
    name = "g_gan"

    def __init__(self, weight: float = 1.0, start_epoch: int = 0, start_step: int | None = None) -> None:
        super().__init__(weight=weight)
        self.start_epoch = int(start_epoch)
        self.start_step = None if start_step is None else int(start_step)

    def is_active(self, epoch: int, global_step: int) -> bool:
        if self.start_step is not None:
            return global_step >= self.start_step
        return epoch >= self.start_epoch

    def compute(self, *, context: dict) -> torch.Tensor:
        fake_pred = context.get("fake_pred")
        device = context["device"]
        dtype = context["dtype"]
        if fake_pred is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return generator_hinge_loss(fake_pred).to(device=device, dtype=dtype)


@LOSS_REGISTRY.register("gan_discriminator")
class GANDiscriminatorLoss(BaseLossComponent):
    name = "d_gan"

    def __init__(self, weight: float = 1.0, start_epoch: int = 0, start_step: int | None = None) -> None:
        super().__init__(weight=weight)
        self.start_epoch = int(start_epoch)
        self.start_step = None if start_step is None else int(start_step)

    def is_active(self, epoch: int, global_step: int) -> bool:
        if self.start_step is not None:
            return global_step >= self.start_step
        return epoch >= self.start_epoch

    def compute(self, *, context: dict) -> torch.Tensor:
        real_pred = context.get("real_pred")
        fake_pred = context.get("fake_pred")
        device = context["device"]
        dtype = context["dtype"]
        if real_pred is None or fake_pred is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return discriminator_hinge_loss(real_pred, fake_pred).to(device=device, dtype=dtype)
