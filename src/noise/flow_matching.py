from __future__ import annotations

import torch

from .endpoint_flow import EndpointFlowNoise
from .registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("flow_matching")
class FlowMatchingNoise(EndpointFlowNoise):
    """Standard flow matching with Gaussian source endpoint."""

    family_key = "flow_matching"
    error_label = "flow-matching"

    def _resolve_source(
        self,
        clean: torch.Tensor,
        device: torch.device,
        *,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del source
        return torch.randn_like(clean, device=device)


@NOISE_REGISTRY.register("residual_flow_matching")
class ResidualFlowMatchingNoise(EndpointFlowNoise):
    """Residual-coupling flow matching using LDCT as the source endpoint."""

    family_key = "residual_flow_matching"
    error_label = "residual-flow-matching"

    def _resolve_source(
        self,
        clean: torch.Tensor,
        device: torch.device,
        *,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            raise ValueError("Residual flow matching requires a source endpoint tensor (e.g. LDCT).")
        return source.to(device=device, dtype=clean.dtype)
