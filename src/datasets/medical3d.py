from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from skimage.transform import resize

from .base import BaseDataset


class Medical3DDataset(BaseDataset):
    """
    Generic 3D medical volume dataset.

    Loads volumetric targets and optional conditioning volumes, normalizes them
    into the canonical image domain, and returns tensors shaped as `(C, D, H, W)`.
    """

    def __init__(
        self,
        file_path: str,
        train: bool = True,
        img_size: int | Tuple[int, int] | Tuple[int, int, int] | None = None,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        norm: bool = True,
        img_datatype=np.float32,
        transforms=None,
        conditioning: bool = False,
        id_key: str | None = "id",
        target_key: str = "target",
        conditioning_key: str | None = "conditioning",
        conditioning_fallback_key: str | None = None,
        split_names: Tuple[str, ...] | None = None,
        split_file: str | Path | None = None,
        use_tensor_cache: bool = True,
        save_tensor_cache: bool = False,
        cache_subdir: str = "cache",
        preprocess_kwargs: dict | None = None,
    ):
        resolved_size = img_size if img_size is not None else volume_size
        super().__init__(
            file_path=file_path,
            train=train,
            img_size=resolved_size,
            norm=norm,
            img_datatype=img_datatype,
            transforms=transforms,
            conditioning=conditioning,
            id_key=id_key,
            target_key=target_key,
            conditioning_key=conditioning_key,
            conditioning_fallback_key=conditioning_fallback_key,
            split_names=split_names,
            split_file=split_file,
            use_tensor_cache=use_tensor_cache,
            save_tensor_cache=save_tensor_cache,
            cache_subdir=cache_subdir,
            preprocess_kwargs=preprocess_kwargs,
        )
        if self.img_size is None or len(self.img_size) != 3:
            raise ValueError("Medical3DDataset requires a 3D volume_size/img_size tuple (D, H, W).")
        self.volume_size = tuple(int(v) for v in self.img_size)

    def preprocess(self, payload: dict) -> np.ndarray:
        volume = payload["Image"] if isinstance(payload, dict) else payload
        volume = np.asarray(volume)
        volume = self._to_channel_first(volume)
        if self.img_size is not None:
            volume = np.stack(
                [
                    resize(channel, self.img_size, preserve_range=True, anti_aliasing=True)
                    for channel in volume
                ],
                axis=0,
            )
        return self.to_image(volume)

    def __getitem__(self, idx, target_resolution: int | None = None):
        sample = super().__getitem__(idx, target_resolution=target_resolution)
        if target_resolution is not None:
            sample["img_size"] = (int(target_resolution), int(target_resolution), int(target_resolution))
        return sample

    def _load_entry(self, entry, item_id):
        resolved = self._resolve_entry_path(entry)
        return super()._load_entry(resolved, item_id)

    def _to_channel_first(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim == 3:
            return np.expand_dims(volume, axis=0)
        if volume.ndim != 4:
            raise ValueError(
                f"Medical3DDataset expects a 3D volume or channelized 4D volume, got shape {tuple(volume.shape)}."
            )
        if volume.shape[0] <= 4:
            return volume
        if volume.shape[-1] <= 4:
            return np.moveaxis(volume, -1, 0)
        raise ValueError(
            "Ambiguous 4D medical volume layout. Expected channel-first [C,D,H,W] "
            "or channel-last [D,H,W,C] with a small channel dimension."
        )

    def _resolve_entry_path(self, entry):
        if isinstance(entry, str):
            path = Path(entry)
            return str(path if path.is_absolute() else self.base_path / path)
        if isinstance(entry, list):
            return [self._resolve_entry_path(item) for item in entry]
        if isinstance(entry, dict):
            resolved = dict(entry)
            if "path" in resolved and resolved["path"] is not None:
                resolved["path"] = self._resolve_entry_path(resolved["path"])
            if isinstance(resolved.get("paths"), list):
                resolved["paths"] = [self._resolve_entry_path(item) for item in resolved["paths"]]
            return resolved
        return entry


__all__ = ["Medical3DDataset"]
