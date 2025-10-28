"""
Abstract base class for modules that consume timestep embeddings alongside the
primary input tensor.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class TimestepBlock(nn.Module, ABC):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:  # pragma: no cover - interface
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        raise NotImplementedError
