"""
Losses and discriminators used for VAE training.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision import models
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - torchvision may be missing in some envs
    _HAS_TORCHVISION = False


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss inspired by LPIPS.

    If torchvision is unavailable the loss gracefully falls back to zero so
    training can proceed without failing hard.
    """

    def __init__(
        self,
        resize: bool = False,
        layers: Tuple[int, ...] = (3, 8, 15, 22),
        layer_weights: Iterable[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.enabled = _HAS_TORCHVISION
        self.resize = resize
        self.layer_weights = list(layer_weights)

        if not self.enabled:
            # Keep a dummy parameter for device placement even when disabled.
            self.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
            return

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features  # type: ignore[attr-defined]
        self.features = vgg.eval()
        self.layers = set(layers)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

        # VGG expects 3-channel inputs; repeat if necessary.
        if recon.shape[1] == 1:
            recon = recon.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.resize:
            recon = F.interpolate(recon, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        loss = 0.0
        weight_iter = iter(self.layer_weights)
        for idx, layer in enumerate(self.features):
            recon = layer(recon)
            target = layer(target)
            if idx in self.layers:
                loss = loss + next(weight_iter, 1.0) * F.l1_loss(recon, target)
        return loss


class PatchDiscriminator(nn.Module):
    """Small PatchGAN-style discriminator used by the VAE."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 8, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def discriminator_hinge_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Standard hinge loss for the discriminator."""
    return torch.mean(F.relu(1.0 - real_pred)) + torch.mean(F.relu(1.0 + fake_pred))


def generator_hinge_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss that encourages fake predictions to be “real”."""
    return -torch.mean(fake_pred)


def vq_regularizer(latents: torch.Tensor) -> torch.Tensor:
    """
    VQ-GAN style regularizer that nudges latents toward zero-mean / unit-variance.

    This is a lightweight surrogate for a full codebook; it penalizes both the
    channel-wise mean and variance drift to prevent arbitrarily scaled latents.
    """
    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    centered = latents - mean
    var = torch.mean(centered.pow(2))
    mean_penalty = torch.mean(mean.pow(2))
    return mean_penalty + var


class FocalLoss(nn.Module):
    """
    Standard focal loss for binary classification (expects logits).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Targets expected in {0,1}; logits are raw model outputs.
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class BCEFocalWrapper(nn.Module):
    """
    Combine BCE with focal loss (both use logits). Useful for reconstructions needing extra focus.
    """

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, focal_weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        self.focal_weight = focal_weight
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        foc = self.focal(logits, targets)
        return bce + self.focal_weight * foc
