from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


def _image_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference image gradients for (B, C, H, W) tensors."""
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dx = x_pad[:, :, 1:-1, 2:] - x_pad[:, :, 1:-1, :-2]
    dy = x_pad[:, :, 2:, 1:-1] - x_pad[:, :, :-2, 1:-1]
    return dx, dy


@LOSS_REGISTRY.register("gradient")
class GradientLoss(BaseLossComponent):
    """Gradient-matching loss to discourage blurred reconstructions."""

    name = "recon_gradient"

    def compute(self, *, context: dict) -> torch.Tensor:
        pred = context["reconstruction_image"]
        target = context["target"]
        p_dx, p_dy = _image_gradient(pred)
        t_dx, t_dy = _image_gradient(target)
        return (p_dx - t_dx).abs().mean() + (p_dy - t_dy).abs().mean()
