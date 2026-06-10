from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a 2-D Gaussian kernel for depthwise SSIM convolutions."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute 1 - SSIM(pred, target) for 2-D image batches in [0, 1].
    """
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"ssim_loss expects 4-D tensors (B,C,H,W), got {pred.dim()} and {target.dim()}.")
    if pred.shape != target.shape:
        raise ValueError(f"ssim_loss expects matching shapes, got {tuple(pred.shape)} vs {tuple(target.shape)}.")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    orig_dtype = pred.dtype
    pred = pred.float()
    target = target.float()

    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (F.conv2d(pred * pred,     kernel, padding=pad, groups=channels) - mu1_sq).clamp(min=0)
    sigma2_sq = (F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq).clamp(min=0)
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return (1.0 - ssim_map.mean()).to(orig_dtype)
