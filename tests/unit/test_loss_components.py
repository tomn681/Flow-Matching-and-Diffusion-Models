import torch

import losses  # noqa: F401 - import for registry side effects
import losses.adversarial  # noqa: F401
import losses.denoising  # noqa: F401
import losses.gradient  # noqa: F401
import losses.perceptual  # noqa: F401
import losses.regularization  # noqa: F401
import losses.reconstruction  # noqa: F401
import losses.ssim  # noqa: F401
from losses.registry import LOSS_REGISTRY
from losses.perceptual import PerceptualLossComponent
from losses.gradient import GradientLoss
from losses.denoising import DenoisingMSELoss
from losses.reconstruction import BCEFocalLoss, BCELoss, FocalLoss, L1Loss, MSELoss
from losses.ssim import SSIMLoss
from nn.losses.ssim import ssim_loss


def test_loss_registry_contains_reconstruction_losses() -> None:
    keys = set(LOSS_REGISTRY.list())
    assert {
        "bce",
        "bce_focal",
        "denoising_mse",
        "focal",
        "gan_discriminator",
        "gan_generator",
        "gradient",
        "kl",
        "l1",
        "mse",
        "perceptual",
        "ssim",
        "vq",
    }.issubset(keys)


def test_reconstruction_components_compute_scalar() -> None:
    pred = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    context = {
        "reconstruction": pred,
        "reconstruction_image": pred,
        "target": target,
    }

    for cls in (L1Loss, MSELoss, BCELoss, FocalLoss, BCEFocalLoss):
        value = cls().compute(context=context)
        assert value.ndim == 0


def test_ssim_and_gradient_components_compute_scalar() -> None:
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    context = {
        "reconstruction_image": pred,
        "target": target,
    }

    for cls in (SSIMLoss, GradientLoss):
        value = cls().compute(context=context)
        assert value.ndim == 0


def test_ssim_zero_for_identical_inputs() -> None:
    image = torch.rand(2, 1, 16, 16)
    loss = SSIMLoss().compute(context={"reconstruction_image": image, "target": image})
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_ssim_loss_stays_finite_for_zero_half_precision_inputs() -> None:
    pred = torch.zeros(2, 1, 16, 16, dtype=torch.float16)
    target = torch.zeros(2, 1, 16, 16, dtype=torch.float16)
    loss = ssim_loss(pred, target)
    assert loss.dtype == torch.float16
    assert torch.isfinite(loss)


def test_ssim_component_is_active_from_configured_epoch() -> None:
    loss = SSIMLoss(weight=1.0, start_epoch=20)
    assert not loss.is_active(epoch=19, global_step=0)
    assert loss.is_active(epoch=20, global_step=0)


def test_perceptual_component_is_active_from_configured_epoch(monkeypatch) -> None:
    class _FakePerceptualLoss:
        def to(self, device):
            return self

        def __call__(self, pred, target):
            return torch.mean(torch.abs(pred - target))

    monkeypatch.setattr("losses.perceptual.PerceptualLoss", lambda **kwargs: _FakePerceptualLoss())
    loss = PerceptualLossComponent(weight=1.0, start_epoch=20)
    assert not loss.is_active(epoch=19, global_step=0)
    assert loss.is_active(epoch=20, global_step=0)


def test_denoising_mse_applies_min_snr_gamma_weighting() -> None:
    pred = torch.tensor([[[[1.0]]], [[[2.0]]]])
    target = torch.zeros_like(pred)
    snr = torch.tensor([10.0, 1.0])
    loss = DenoisingMSELoss(min_snr_gamma=5.0).compute(
        context={
            "pred": pred,
            "target": target,
            "snr": snr,
            "prediction_type": "epsilon",
        }
    )
    expected = ((1.0**2) * (5.0 / 10.0) + (2.0**2) * (1.0 / 1.0)) / 2.0
    assert torch.isclose(loss, torch.tensor(expected))
