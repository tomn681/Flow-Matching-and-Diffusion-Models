"""Feature-space objectives for VAE intermediate supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.vae.intermediate import IntermediateFeaturePair
from .base import BaseLossComponent
from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("intermediate_feature")
class IntermediateFeatureLoss(BaseLossComponent):
    """Compare one aligned encoder/decoder feature pair."""

    def __init__(self, *, level: int, loss: str = "l1", weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.level = int(level)
        self.loss = str(loss).strip().lower()
        if self.loss not in {"l1", "mse", "smooth_l1", "cosine"}:
            raise ValueError(f"Unsupported intermediate feature loss '{self.loss}'.")
        self.name = f"intermediate_l{self.level + 1}_{self.loss}"

    def compute(self, *, context: dict) -> torch.Tensor:
        features = context.get("intermediate_features")
        if not isinstance(features, dict) or self.level not in features:
            raise RuntimeError(f"Intermediate features for VAE level {self.level + 1} were not provided.")
        pair = features[self.level]
        if not isinstance(pair, IntermediateFeaturePair):
            raise TypeError(
                f"Expected IntermediateFeaturePair at level {self.level + 1}, got {type(pair).__name__}."
            )
        if self.loss == "l1":
            return F.l1_loss(pair.decoder, pair.encoder)
        if self.loss == "mse":
            return F.mse_loss(pair.decoder, pair.encoder)
        if self.loss == "smooth_l1":
            return F.smooth_l1_loss(pair.decoder, pair.encoder)
        decoder = pair.decoder.flatten(1)
        encoder = pair.encoder.flatten(1)
        return (1.0 - F.cosine_similarity(decoder, encoder, dim=1)).mean()


__all__ = ["IntermediateFeatureLoss"]
