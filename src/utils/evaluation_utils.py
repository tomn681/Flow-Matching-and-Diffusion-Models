from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nn.losses.ssim import _gaussian_kernel, ssim_loss

from .indexing_utils import select_visual_indices


def latent_shape(vae_cfg: dict) -> tuple[int, ...]:
    spatial_dims = vae_cfg.get("spatial_dims", 2)
    latent_channels = vae_cfg.get("embed_dim", vae_cfg.get("latent_channels", vae_cfg.get("z_channels")))
    if latent_channels is None:
        raise KeyError("VAE config must define one of: embed_dim, latent_channels, or z_channels.")
    resolution = vae_cfg["resolution"]
    down_channels = vae_cfg.get("down_channels")
    channels = vae_cfg.get("channels")
    if down_channels is not None:
        factor = 2 ** (len(tuple(down_channels)) - 1)
    elif channels is not None:
        factor = 2 ** (len(tuple(channels)) - 1)
    else:
        ch_mult = tuple(vae_cfg["ch_mult"])
        factor = 2 ** (len(ch_mult) - 1)
    base_size = resolution // factor
    if spatial_dims == 3:
        return (latent_channels, base_size, base_size, base_size)
    if spatial_dims == 1:
        return (latent_channels, base_size)
    return (latent_channels, base_size, base_size)


def make_grid(tensor: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    n, c, h, w = tensor.shape
    if n < rows * cols:
        raise ValueError(f"Need at least {rows*cols} images to build the grid, found {n}")
    tensor = tensor[: rows * cols]
    if c == 1:
        tensor = tensor.expand(-1, 3, h, w)
        c = 3
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.reshape(rows, cols, c, h, w)
    tensor = tensor.permute(2, 0, 3, 1, 4).contiguous()
    grid = tensor.reshape(c, rows * h, cols * w)
    grid_np = (grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    grid_np = np.transpose(grid_np, (1, 2, 0))
    return grid_np


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    logging.info("Saved grid: %s", path)


def prepare_eval_batch(ds, count: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
    if ds is None or len(ds) == 0:
        raise RuntimeError("Dataset is empty; cannot prepare evaluation batch.")
    indices = select_visual_indices(ds, count, seed=seed)
    tensors = [ds[i]["target"] for i in indices]
    if not tensors:
        raise RuntimeError("Failed to collect evaluation samples.")
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


def compute_ssim_sample(pred: torch.Tensor, tgt: torch.Tensor, ssim_fn) -> float | None:
    """
    Compute SSIM for one sample in channel-first layout.
    Supports N-dimensional spatial tensors by averaging per-channel SSIM.
    """
    if pred.shape != tgt.shape:
        return None
    pred = pred.detach()
    tgt = tgt.detach()
    if pred.device.type != "cpu":
        pred = pred.cpu()
    if tgt.device.type != "cpu":
        tgt = tgt.cpu()
    if pred.dtype != torch.float32:
        pred = pred.float()
    if tgt.dtype != torch.float32:
        tgt = tgt.float()

    if pred.ndim < 2:
        return None

    if pred.ndim == 2:
        return float(ssim_fn(pred.numpy(), tgt.numpy(), channel_axis=None, data_range=1.0))

    if pred.ndim == 3:
        return float(1.0 - ssim_loss(pred.unsqueeze(0), tgt.unsqueeze(0)).item())

    # Assume channel-first for ndim >= 3 and average SSIM per channel.
    # Each channel slice may be 2D (image), 3D (volume/video), or higher.
    channel_scores = []
    for ch in range(pred.shape[0]):
        p = pred[ch].numpy()
        t = tgt[ch].numpy()
        if p.ndim < 2:
            continue
        channel_scores.append(float(ssim_fn(p, t, channel_axis=None, data_range=1.0)))
    if not channel_scores:
        return None
    return float(np.mean(channel_scores))


def compute_ssim_batch(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute per-sample SSIM scores for 2-D image batches in [0, 1].

    Returns:
        Tensor of shape [B] with SSIM scores in [0, 1].
    """
    if pred.dim() != 4 or tgt.dim() != 4:
        raise ValueError(f"compute_ssim_batch expects 4-D tensors (B,C,H,W), got {pred.dim()} and {tgt.dim()}.")
    if pred.shape != tgt.shape:
        raise ValueError(f"compute_ssim_batch expects matching shapes, got {tuple(pred.shape)} vs {tuple(tgt.shape)}.")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    pred = pred.float()
    tgt = tgt.float()
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(tgt, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq).clamp(min=0)
    sigma2_sq = (F.conv2d(tgt * tgt, kernel, padding=pad, groups=channels) - mu2_sq).clamp(min=0)
    sigma12 = F.conv2d(pred * tgt, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean(dim=(1, 2, 3))
