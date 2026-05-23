from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentCacheDataset(Dataset):
    """Dataset wrapper that reads precomputed latent tensors from `.pt` files."""

    def __init__(self, cache_dir: str | Path, split: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Latent cache directory not found: {self.cache_dir}")
        if split:
            split_dir = self.cache_dir / str(split)
            if not split_dir.exists():
                raise FileNotFoundError(f"Latent cache split directory not found: {split_dir}")
            self.files = sorted(split_dir.glob("*.pt"))
        else:
            direct = sorted(self.cache_dir.glob("*.pt"))
            self.files = direct if direct else sorted(self.cache_dir.rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt latent files found in: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        payload = torch.load(self.files[idx], map_location="cpu")
        if isinstance(payload, dict):
            if "target" not in payload:
                raise KeyError(f"Latent file '{self.files[idx]}' must contain a 'target' key.")
            out = {"target": payload["target"]}
            if payload.get("image") is not None:
                out["image"] = payload["image"]
            return out
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Unsupported latent payload type '{type(payload).__name__}' in '{self.files[idx]}'.")
        return {"target": payload}


__all__ = ["LatentCacheDataset"]
