"""
Focal Frequency Loss for reconstruction-focused experiments.
"""

from __future__ import annotations

import torch

from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("focal_frequency")
class FocalFrequencyLoss(BaseLossComponent):
    name = "recon_ffl"

    def __init__(self, weight: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.alpha = float(alpha)

    def compute(self, *, context: dict) -> torch.Tensor:
        pred = context["reconstruction_image"].float()
        target = context["target"].float()
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")
        diff_real = pred_fft.real - target_fft.real
        diff_imag = pred_fft.imag - target_fft.imag
        freq_error = (diff_real.square() + diff_imag.square()).sqrt()
        with torch.no_grad():
            weight_matrix = freq_error.pow(self.alpha)
            weight_matrix = weight_matrix / (weight_matrix.mean() + 1e-8)
        loss = (weight_matrix * (diff_real.square() + diff_imag.square())).mean()
        return loss.to(dtype=context["reconstruction_image"].dtype)
