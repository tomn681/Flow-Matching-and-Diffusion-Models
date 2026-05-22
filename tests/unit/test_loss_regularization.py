from __future__ import annotations

import pytest
import torch

from losses.regularization import KLLoss, VQLoss


class _FakePosterior:
    def __init__(self, kl_value: float):
        self._kl = kl_value

    def kl(self) -> torch.Tensor:
        return torch.tensor([self._kl, self._kl], dtype=torch.float32)


def test_kl_loss_with_posterior() -> None:
    loss = KLLoss(weight=1.0)
    ctx = {
        "posterior": _FakePosterior(0.5),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.ndim == 0
    assert value.item() == pytest.approx(0.5)


def test_kl_loss_without_posterior_returns_zero() -> None:
    loss = KLLoss(weight=1.0)
    ctx = {
        "posterior": None,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.item() == 0.0


def test_vq_loss_with_codebook_loss() -> None:
    loss = VQLoss(weight=1.0)
    ctx = {
        "codebook_loss": torch.tensor(0.25),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.item() == pytest.approx(0.25)


def test_vq_loss_without_codebook_returns_zero() -> None:
    loss = VQLoss(weight=1.0)
    ctx = {
        "codebook_loss": None,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.item() == 0.0

