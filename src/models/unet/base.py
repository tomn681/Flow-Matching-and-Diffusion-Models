from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseUNetND(nn.Module, ABC):
    """
    Common interface for UNet generators used by diffusion/flow-matching code.
    """

    def _normalize_timesteps(self, t, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        if t.ndim == 0:
            t = t[None].to(x.device)
        return t.expand(x.shape[0]).to(x.device)

    def _prepare_input(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        context_ca: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return x

    @abstractmethod
    def _build_time_embedding(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _run_network(self, x: torch.Tensor, emb: torch.Tensor, context_ca: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        context: Optional[torch.Tensor] = None,
        context_ca: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self._prepare_input(x, context, context_ca)
        t = self._normalize_timesteps(t, x)
        emb = self._build_time_embedding(t, x)
        y = self._run_network(x, emb, context_ca)
        return self._postprocess_output(y)
