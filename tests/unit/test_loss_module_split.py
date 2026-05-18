import torch

from nn.losses import (
    PerceptualLoss,
    PatchDiscriminator,
    bce_focal_loss,
    discriminator_hinge_loss,
    focal_loss,
    generator_hinge_loss,
    vq_regularizer,
)
from nn.losses.adversarial import PatchDiscriminator as PatchDiscNew
from nn.losses.perceptual import PerceptualLoss as PerceptualNew
from nn.losses.reconstruction import bce_focal_loss as bce_focal_new
from nn.losses.regularization import vq_regularizer as vq_regularizer_new
from nn.losses.vae import (
    PatchDiscriminator as PatchDiscShim,
    PerceptualLoss as PerceptualShim,
)


def test_split_exports_are_available() -> None:
    assert PerceptualLoss is PerceptualNew
    assert PatchDiscriminator is PatchDiscNew
    assert bce_focal_loss is bce_focal_new
    assert vq_regularizer is vq_regularizer_new


def test_vae_shim_reexports_symbols() -> None:
    assert PatchDiscShim is PatchDiscNew
    assert PerceptualShim is PerceptualNew


def test_loss_functions_smoke() -> None:
    logits = torch.randn(2, 1, 4, 4)
    targets = torch.rand(2, 1, 4, 4)
    real = torch.randn(2, 1, 2, 2)
    fake = torch.randn(2, 1, 2, 2)

    assert focal_loss(logits, targets).ndim == 0
    assert bce_focal_loss(logits, targets).ndim == 0
    assert discriminator_hinge_loss(real, fake).ndim == 0
    assert generator_hinge_loss(fake).ndim == 0
    assert vq_regularizer(torch.randn(2, 4, 8, 8)).ndim == 0
