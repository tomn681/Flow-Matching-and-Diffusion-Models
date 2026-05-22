import torch

import losses  # noqa: F401 - import for registry side effects
import losses.adversarial  # noqa: F401
import losses.perceptual  # noqa: F401
import losses.regularization  # noqa: F401
import losses.reconstruction  # noqa: F401
from losses.registry import LOSS_REGISTRY
from losses.reconstruction import BCEFocalLoss, BCELoss, FocalLoss, L1Loss, MSELoss


def test_loss_registry_contains_reconstruction_losses() -> None:
    keys = set(LOSS_REGISTRY.list())
    assert {
        "bce",
        "bce_focal",
        "focal",
        "gan_discriminator",
        "gan_generator",
        "kl",
        "l1",
        "mse",
        "perceptual",
        "vq",
    }.issubset(keys)


def test_reconstruction_components_compute_scalar() -> None:
    pred = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)

    for cls in (L1Loss, MSELoss, BCELoss, FocalLoss, BCEFocalLoss):
        value = cls().compute(pred, target)
        assert value.ndim == 0
