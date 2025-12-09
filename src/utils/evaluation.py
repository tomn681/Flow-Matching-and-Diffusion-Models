from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image


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
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
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


def prepare_eval_batch(ds, count: int, device: torch.device) -> torch.Tensor:
    if ds is None or len(ds) == 0:
        raise RuntimeError("Dataset is empty; cannot prepare evaluation batch.")
    tensors = [ds[i]["target"] for i in range(min(len(ds), count))]
    if not tensors:
        raise RuntimeError("Failed to collect evaluation samples.")
    batch = torch.stack(tensors, dim=0).to(device)
    batch = batch * 2.0 - 1.0
    return batch
