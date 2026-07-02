"""
Runtime helpers for dataset-backed caching, batching, and tensor export.

This module holds the operational pieces that used to live in `dataset_utils.py`
so dataset config resolution and dataset runtime IO are no longer mixed together.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def cache_path_for_entry(
    base_path: Path,
    cache_root: Path,
    entry,
    split_index: int | None = None,
    split_count: int = 1,
) -> Path | None:
    """Build the stable cache/output path for a dataset entry."""
    if cache_root is None:
        return None
    if isinstance(entry, list):
        if not entry:
            return None
        base = entry[0]
    elif isinstance(entry, dict):
        base = entry.get("path")
        if base is None and isinstance(entry.get("paths"), (list, tuple)) and entry["paths"]:
            base = entry["paths"][0]
    else:
        base = entry

    if base is None:
        return None
    entry_path = Path(str(base))
    if entry_path.is_absolute():
        try:
            rel = entry_path.relative_to(base_path)
            stable_key = None
        except Exception:
            rel = Path(entry_path.name)
            stable_key = hashlib.sha1(str(entry_path).encode("utf-8")).hexdigest()[:12]
    else:
        rel = entry_path
        stable_key = None
    stem = Path(rel).stem
    parent = Path(rel).parent
    if stable_key is not None:
        stem = f"{stem}_{stable_key}"
    if split_count > 1 and split_index is not None:
        filename = f"{stem}_split_{split_index}.pt"
    else:
        filename = f"{stem}.pt"
    return cache_root / parent / filename


def save_tensor_cache(tensor, cache_path: Path) -> None:
    """Atomically save a tensor to a cache/output path."""
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    try:
        with open(tmp_path, "rb+") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(tmp_path, cache_path)


def iter_batches(dataset, batch_size: int, indices: list[int] | None = None):
    """Yield index lists and sample batches from a dataset."""
    selected = list(range(len(dataset))) if indices is None else list(indices)
    total = len(selected)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_indices = selected[start:end]
        samples = [dataset[i] for i in batch_indices]
        yield batch_indices, samples


def save_output_tensor(dataset, row: dict, key: str, tensor, output_root: Path) -> None:
    """Save a tensor under an output root using the dataset cache layout."""
    entry = row.get(key)
    split_index, split_count = dataset._cache_info(entry, row, key)
    out_path = cache_path_for_entry(dataset.base_path, output_root, entry, split_index, split_count)
    if out_path is None:
        return
    writer = getattr(dataset, "save_output", None)
    if callable(writer):
        writer(row=row, key=key, tensor=tensor, output_root=output_root)
        return
    save_tensor_cache(tensor, out_path)


def save_artifact_image(dataset, row: dict, key: str, tensor, output_root: Path) -> None:
    """Save a derived artifact as PNG while reusing dataset naming/layout."""
    entry = row.get(key)
    split_index, split_count = dataset._cache_info(entry, row, key)
    out_path = cache_path_for_entry(dataset.base_path, output_root, entry, split_index, split_count)
    if out_path is None:
        return
    png_path = out_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    arr = torch.as_tensor(tensor).detach().cpu().float()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        img = (arr.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(img, mode="L").save(png_path)
        return
    if arr.ndim == 3 and arr.shape[0] == 1:
        img = (arr[0].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(img, mode="L").save(png_path)
        return
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = (arr.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(png_path)
        return
    save_tensor_cache(arr, out_path)


def to_2d_image(arr: torch.Tensor) -> np.ndarray | None:
    """
    Convert common tensor layouts to uint8 grayscale image if possible.
    Supports [H,W], [1,H,W], [C,H,W] with C in {1,3}.
    """
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3 and arr.shape[0] == 1:
        img = arr[0]
    elif arr.ndim == 3 and arr.shape[0] in (3,):
        img = arr.mean(dim=0)
    else:
        return None
    img = img.clamp(0.0, 1.0).numpy()
    return (img * 255.0).round().astype(np.uint8)


__all__ = [
    "cache_path_for_entry",
    "save_tensor_cache",
    "iter_batches",
    "save_output_tensor",
    "save_artifact_image",
    "to_2d_image",
]
