from __future__ import annotations

import torch

from .endpoint_flow import EndpointFlowNoise
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("rectified_flow")
class RectifiedFlowNoise(EndpointFlowNoise):
    """Rectified flow with Gaussian source endpoint."""

    family_key = "rectified_flow"
    error_label = "rectified-flow"

    def _resolve_source(
        self,
        clean: torch.Tensor,
        device: torch.device,
        *,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del source
        return torch.randn_like(clean, device=device)


@NOISE_REGISTRY.register("residual_rectified_flow")
class ResidualRectifiedFlowNoise(EndpointFlowNoise):
    """Residual-coupling rectified flow using LDCT as the source endpoint."""

    family_key = "residual_rectified_flow"
    error_label = "residual-rectified-flow"

    def _resolve_source(
        self,
        clean: torch.Tensor,
        device: torch.device,
        *,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            raise ValueError("Residual rectified flow requires a source endpoint tensor (e.g. LDCT).")
        return source.to(device=device, dtype=clean.dtype)
