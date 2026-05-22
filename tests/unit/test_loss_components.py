import torch

from losses.registry import LOSS_REGISTRY
from losses.reconstruction import BCEFocalLoss, BCELoss, FocalLoss, L1Loss, MSELoss


def test_loss_registry_contains_reconstruction_losses() -> None:
    assert LOSS_REGISTRY.list() == ["bce", "bce_focal", "focal", "l1", "mse"]


def test_reconstruction_components_compute_scalar() -> None:
    pred = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)

    for cls in (L1Loss, MSELoss, BCELoss, FocalLoss, BCEFocalLoss):
        value = cls().compute(pred, target)
        assert value.ndim == 0
