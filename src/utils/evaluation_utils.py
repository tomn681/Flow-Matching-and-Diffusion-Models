from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from utils.utils import select_visual_indices


def latent_shape(vae_cfg: dict) -> tuple[int, ...]:
    spatial_dims = vae_cfg.get("spatial_dims", 2)
    embed_dim = vae_cfg["embed_dim"]
    resolution = vae_cfg["resolution"]
    down_channels = vae_cfg.get("down_channels")
    if down_channels is not None:
        factor = 2 ** (len(tuple(down_channels)) - 1)
    else:
        ch_mult = tuple(vae_cfg["ch_mult"])
        factor = 2 ** (len(ch_mult) - 1)
    base_size = resolution // factor
    if spatial_dims == 3:
        return (embed_dim, base_size, base_size, base_size)
    if spatial_dims == 1:
        return (embed_dim, base_size)
    return (embed_dim, base_size, base_size)


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
    pred = pred.detach().cpu().float()
    tgt = tgt.detach().cpu().float()

    if pred.ndim < 2:
        return None

    if pred.ndim == 2:
        return float(ssim_fn(pred.numpy(), tgt.numpy(), channel_axis=None, data_range=1.0))

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
