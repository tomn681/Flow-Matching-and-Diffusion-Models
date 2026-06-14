from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

import utils


class LatentCacheDataset(Dataset):
    """Dataset wrapper that reads precomputed latent tensors from `.pt` files."""

    def __init__(self, cache_dir: str | Path, split: str | None = None, *, preload: bool = False) -> None:
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
        self._preloaded = [self._load_payload(path) for path in self.files] if preload else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        payload = self._preloaded[idx] if self._preloaded is not None else self._load_payload(self.files[idx])
        return self._normalize_payload(payload, self.files[idx])

    @staticmethod
    def _load_payload(path: Path):
        return utils.safe_torch_load(path, map_location="cpu")

    @staticmethod
    def _normalize_payload(payload, path: Path) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            if "target" not in payload:
                raise KeyError(f"Latent file '{path}' must contain a 'target' key.")
            out = {"target": payload["target"]}
            if payload.get("image") is not None:
                out["image"] = payload["image"]
            return out
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Unsupported latent payload type '{type(payload).__name__}' in '{path}'.")
        return {"target": payload}


__all__ = ["LatentCacheDataset"]
