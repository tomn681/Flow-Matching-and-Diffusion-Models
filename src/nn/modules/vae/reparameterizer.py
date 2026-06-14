"""
Reparameterization helpers for VAEs.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch


class DiagonalGaussian:
    """
    q(z|x) diagonal with sampling, KL, and NLL utilities.
    `parameters` = (B, 2*C, H, W) with [mu, logvar] along channels.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        mu, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        self.mu = mu
        self.logvar = logvar
        self.deter = deterministic
        self.device = parameters.device

        if deterministic:
            self.std = torch.zeros_like(mu, device=self.device)
            self.var = torch.zeros_like(mu, device=self.device)
        else:
            self.std = torch.exp(0.5 * logvar)
            self.var = torch.exp(logvar)

    def sample(self) -> torch.Tensor:
        if self.deter:
            return self.mu
        return self.mu + self.std * torch.randn_like(self.mu, device=self.device)

    def mode(self) -> torch.Tensor:
        return self.mu

    def _resolve_reduce_dims(self, reduce_dims: Optional[Iterable[int]]) -> tuple[int, ...]:
        if reduce_dims is None:
            return tuple(range(1, self.mu.ndim))
        return tuple(reduce_dims)

    def kl(self, other: Optional["DiagonalGaussian"] = None, reduce_dims: Optional[Iterable[int]] = None) -> torch.Tensor:
        reduce_dims = self._resolve_reduce_dims(reduce_dims)
        if self.deter:
            return torch.zeros(self.mu.size(0), device=self.device, dtype=self.mu.dtype)
        mu = self.mu.float()
        var = self.var.float()
        logvar = self.logvar.float()
        if other is None:
            out = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - logvar, dim=reduce_dims)
            return out.to(dtype=self.mu.dtype)
        other_mu = other.mu.float()
        other_var = other.var.float()
        other_logvar = other.logvar.float()
        out = 0.5 * torch.sum(
            (mu - other_mu).pow(2) / other_var + var / other_var - 1.0 - logvar + other_logvar,
            dim=reduce_dims,
        )
        return out.to(dtype=self.mu.dtype)

    def nll(self, x: torch.Tensor, reduce_dims: Optional[Iterable[int]] = None) -> torch.Tensor:
        reduce_dims = self._resolve_reduce_dims(reduce_dims)
        logtwopi = math.log(2.0 * math.pi)
        x_f = x.float()
        mu = self.mu.float()
        logvar = self.logvar.float()
        var = self.var.float()
        out = 0.5 * torch.sum(logtwopi + logvar + (x_f - mu).pow(2) / var, dim=reduce_dims)
        return out.to(dtype=x.dtype)
