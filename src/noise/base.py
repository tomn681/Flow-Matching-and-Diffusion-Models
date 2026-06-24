from __future__ import annotations

import abc
from typing import Any

import torch

from core.types import NoisyBatch


class BaseNoiseProcess(abc.ABC):
    """Framework base class for training-time noising processes."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    @abc.abstractmethod
    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        raise NotImplementedError
