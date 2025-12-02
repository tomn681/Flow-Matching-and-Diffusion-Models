"""
Distribution helpers used by the VAE modules.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch


class DiagonalGaussian:
    """
    q(z|x) diagonal con helpers (sample, mode, KL, NLL).
    `parameters` = (B, 2*C, H, W) con [mu, logvar] en el eje de canales.
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

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        if self.deter:
            return self.mu
        return self.mu + self.std * torch.randn_like(self.mu, device=self.device)

    def mode(self) -> torch.Tensor:
        return self.mu

    def kl(self, other: Optional["DiagonalGaussian"] = None, reduce_dims: Iterable[int] = (1, 2, 3)) -> torch.Tensor:
        if self.deter:
            return torch.tensor([0.0], device=self.device)
        if other is None:
            return 0.5 * torch.sum(self.mu.pow(2) + self.var - 1.0 - self.logvar, dim=reduce_dims)
        return 0.5 * torch.sum(
            (self.mu - other.mu).pow(2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=reduce_dims,
        )

    def nll(self, x: torch.Tensor, reduce_dims: Iterable[int] = (1, 2, 3)) -> torch.Tensor:
        logtwopi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + (x - self.mu).pow(2) / self.var, dim=reduce_dims)
