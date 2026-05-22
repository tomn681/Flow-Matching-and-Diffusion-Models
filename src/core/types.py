from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


def unwrap_model_prediction(pred: Any) -> torch.Tensor:
    """Extract a tensor prediction from heterogeneous model outputs."""
    if isinstance(pred, (list, tuple)):
        return pred[0]
    sample = getattr(pred, "sample", None)
    if sample is not None:
        return sample
    return pred


@dataclass
class ModelOutput:
    """Unified return type for generative model forward passes."""

    reconstruction: torch.Tensor
    posterior: Optional[Any] = None
    codebook_loss: Optional[torch.Tensor] = None
    auxiliary: dict = field(default_factory=dict)


@dataclass
class NoisyBatch:
    """Result of corrupting clean data for training."""

    noisy: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor


@dataclass
class TrainingState:
    """Snapshot of training state for checkpointing and callbacks."""

    epoch: int
    global_step: int
    model_state: dict
    optimizer_state: dict
    metrics: dict
    config: Any = None
    extra: dict = field(default_factory=dict)
