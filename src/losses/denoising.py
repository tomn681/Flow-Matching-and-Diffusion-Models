from __future__ import annotations

import torch

from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


def _reduce_per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per_elem = (pred - target).pow(2)
    reduce_dims = tuple(range(1, per_elem.ndim))
    if not reduce_dims:
        return per_elem
    return per_elem.mean(dim=reduce_dims)


@LOSS_REGISTRY.register("denoising_mse")
class DenoisingMSELoss(BaseLossComponent):
    """Per-sample MSE with optional Min-SNR-gamma weighting."""

    name = "denoise_mse"

    def __init__(self, weight: float = 1.0, *, min_snr_gamma: float = 0.0) -> None:
        super().__init__(weight=weight)
        self.min_snr_gamma = float(min_snr_gamma)

    @staticmethod
    def _prediction_weight(
        *,
        snr: torch.Tensor,
        gamma: float,
        prediction_type: str,
    ) -> torch.Tensor:
        clipped = torch.minimum(snr, torch.full_like(snr, gamma))
        if prediction_type == "epsilon":
            return clipped / snr.clamp_min(1e-8)
        if prediction_type == "v_prediction":
            return clipped / (snr + 1.0)
        if prediction_type == "sample":
            return clipped
        return torch.ones_like(snr)

    def compute(self, *, context: dict) -> torch.Tensor:
        pred = context["pred"]
        target = context["target"]
        per_sample = _reduce_per_sample_mse(pred, target)
        snr = context.get("snr")
        prediction_type = str(context.get("prediction_type", "epsilon")).lower()
        if self.min_snr_gamma > 0.0 and isinstance(snr, torch.Tensor):
            weights = self._prediction_weight(
                snr=snr.to(device=per_sample.device, dtype=per_sample.dtype),
                gamma=self.min_snr_gamma,
                prediction_type=prediction_type,
            )
            per_sample = per_sample * weights
        return per_sample.mean()

