from __future__ import annotations

import math

import pytest
import torch

from nn.modules.vae.reparameterizer import DiagonalGaussian


def test_diagonal_gaussian_kl_reduces_across_all_non_batch_dims_for_3d_posteriors() -> None:
    moments = torch.zeros(2, 8, 3, 4, 5)
    posterior = DiagonalGaussian(moments)

    kl = posterior.kl()

    assert kl.shape == (2,)
    assert torch.allclose(kl, torch.zeros(2))


def test_diagonal_gaussian_nll_reduces_across_all_non_batch_dims_for_3d_posteriors() -> None:
    moments = torch.zeros(2, 8, 3, 4, 5)
    posterior = DiagonalGaussian(moments)
    x = torch.zeros_like(posterior.mu)

    nll = posterior.nll(x)

    expected = 0.5 * posterior.mu[0].numel() * math.log(2.0 * math.pi)
    assert nll.shape == (2,)
    assert nll[0].item() == pytest.approx(expected)
    assert nll[1].item() == pytest.approx(expected)


def test_diagonal_gaussian_deterministic_kl_returns_per_batch_zeros() -> None:
    moments = torch.zeros(3, 8, 4, 4)
    posterior = DiagonalGaussian(moments, deterministic=True)

    kl = posterior.kl()

    assert kl.shape == (3,)
    assert torch.allclose(kl, torch.zeros(3))
