from __future__ import annotations

import torch

from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("kl")
class KLLoss(BaseLossComponent):
    name = "kl"

    def compute(self, *, context: dict) -> torch.Tensor:
        posterior = context.get("posterior")
        device = context["device"]
        dtype = context["dtype"]
        if posterior is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return posterior.kl().mean()


@LOSS_REGISTRY.register("vq")
class VQLoss(BaseLossComponent):
    name = "vq"

    def compute(self, *, context: dict) -> torch.Tensor:
        codebook_loss = context.get("codebook_loss")
        device = context["device"]
        dtype = context["dtype"]
        if codebook_loss is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return codebook_loss.to(device=device, dtype=dtype)
