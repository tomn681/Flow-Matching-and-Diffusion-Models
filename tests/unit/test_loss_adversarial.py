from __future__ import annotations

import torch

from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss


def test_gan_generator_is_active_epoch_based() -> None:
    loss = GANGeneratorLoss(weight=0.5, start_epoch=5)
    assert not loss.is_active(epoch=3, global_step=0)
    assert loss.is_active(epoch=5, global_step=0)
    assert loss.is_active(epoch=10, global_step=0)


def test_gan_generator_is_active_step_based() -> None:
    loss = GANGeneratorLoss(weight=0.5, start_epoch=0, start_step=100)
    assert not loss.is_active(epoch=0, global_step=50)
    assert loss.is_active(epoch=0, global_step=100)
    assert loss.is_active(epoch=99, global_step=200)


def test_gan_generator_compute_with_fake_pred() -> None:
    loss = GANGeneratorLoss(weight=1.0)
    fake_pred = torch.randn(4, 1, 4, 4)
    ctx = {
        "fake_pred": fake_pred,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_gan_generator_compute_without_fake_pred_returns_zero() -> None:
    loss = GANGeneratorLoss(weight=1.0)
    ctx = {
        "fake_pred": None,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.item() == 0.0


def test_gan_discriminator_compute_with_preds() -> None:
    loss = GANDiscriminatorLoss(weight=1.0)
    real_pred = torch.randn(4, 1, 4, 4)
    fake_pred = torch.randn(4, 1, 4, 4)
    ctx = {
        "real_pred": real_pred,
        "fake_pred": fake_pred,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_gan_discriminator_compute_without_preds_returns_zero() -> None:
    loss = GANDiscriminatorLoss(weight=1.0)
    ctx = {
        "real_pred": None,
        "fake_pred": None,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = loss.compute(context=ctx)
    assert value.item() == 0.0


def test_gan_discriminator_is_active_matches_generator() -> None:
    g = GANGeneratorLoss(weight=1.0, start_epoch=3, start_step=None)
    d = GANDiscriminatorLoss(weight=1.0, start_epoch=3, start_step=None)
    for epoch in range(6):
        assert g.is_active(epoch=epoch, global_step=0) == d.is_active(epoch=epoch, global_step=0)

