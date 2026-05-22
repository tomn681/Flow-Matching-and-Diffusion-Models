from __future__ import annotations

import torch

from .base import BaseLossComponent


class KLLoss(BaseLossComponent):
    name = "kl"

    def compute(self, prediction: torch.Tensor, target: torch.Tensor, **context) -> torch.Tensor:
        posterior = context.get("posterior")
        if posterior is None:
            return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        return posterior.kl().mean()


class VQLoss(BaseLossComponent):
    name = "vq"

    def compute(self, prediction: torch.Tensor, target: torch.Tensor, **context) -> torch.Tensor:
        codebook_loss = context.get("codebook_loss")
        if codebook_loss is None:
            return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        return codebook_loss.to(device=prediction.device, dtype=prediction.dtype)
