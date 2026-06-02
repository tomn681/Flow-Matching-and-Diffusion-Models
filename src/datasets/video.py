from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.transform import resize

from utils import load_image

from .base import BaseDataset


class VideoDataset(BaseDataset):
    """Temporal clip dataset for video-like frame sequences.

    The split file is expected to contain one row per frame with at least:
    - a video/group identifier column
    - a target frame path column
    - optionally a conditioning frame path column

    Samples are built as sliding clips of length `clip_length`, where adjacent
    frames inside each clip are separated by `frame_stride`.
    Returned tensors follow the shape `(C, T, H, W)`.
    """

    def __init__(
        self,
        file_path: str,
        train: bool = True,
        img_size: int | Tuple[int, int] | Tuple[int, int, int] | None = None,
        clip_length: int = 16,
        frame_stride: int = 1,
        norm: bool = True,
        img_datatype=np.float32,
        transforms=None,
        conditioning: bool = False,
        id_key: str = "video_id",
        target_key: str = "target",
        conditioning_key: str | None = "conditioning",
        conditioning_fallback_key: str | None = None,
        frame_index_key: str | None = None,
        split_names: Tuple[str, ...] | None = None,
        split_file: str | Path | None = None,
        use_tensor_cache: bool = True,
        save_tensor_cache: bool = False,
        cache_subdir: str = "cache",
    ) -> None:
        if int(clip_length) <= 0:
            raise ValueError("clip_length must be > 0.")
        if int(frame_stride) <= 0:
            raise ValueError("frame_stride must be > 0.")

        super().__init__(
            file_path=file_path,
            train=train,
            img_size=img_size,
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
        )
        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.frame_index_key = frame_index_key
        self._build_video_index(split_names=split_names)

    def _build_video_index(self, split_names: Tuple[str, ...] | None) -> None:
        df = self._read_split_file(self.data_root, names=split_names)
        required_cols = [self.id_key, self.target_key]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if self.frame_index_key is not None and self.frame_index_key not in df.columns:
            raise KeyError(f"frame_index_key '{self.frame_index_key}' is missing from annotations.")

        records: list[dict] = []
        grouped = df.groupby(self.id_key, sort=False)
        for video_id, group in grouped:
            if self.frame_index_key is not None:
                group = group.sort_values(self.frame_index_key, kind="stable")

            rows = group.to_dict("records")
            max_start = len(rows) - (self.clip_length - 1) * self.frame_stride
            if max_start <= 0:
                continue

            split_count = max_start
            for start in range(max_start):
                indices = [start + offset * self.frame_stride for offset in range(self.clip_length)]
                target_paths = [rows[idx][self.target_key] for idx in indices]
                target_entry = {
                    "paths": target_paths,
                    "split_index": start,
                    "split_count": split_count,
                    "window": self.clip_length,
                }

                record = {
                    self.id_key: video_id,
                    self.target_key: target_entry,
                    f"{self.target_key}__split_index": start,
                    f"{self.target_key}__split_count": split_count,
                }

                if self.conditioning_key is not None and self.conditioning_key in group.columns:
                    conditioning_paths = [rows[idx].get(self.conditioning_key) for idx in indices]
                    if not any(self._is_missing_entry(path) for path in conditioning_paths):
                        record[self.conditioning_key] = {
                            "paths": conditioning_paths,
                            "split_index": start,
                            "split_count": split_count,
                            "window": self.clip_length,
                        }
                        record[f"{self.conditioning_key}__split_index"] = start
                        record[f"{self.conditioning_key}__split_count"] = split_count

                if self.conditioning_fallback_key is not None and self.conditioning_fallback_key in group.columns:
                    fallback_paths = [rows[idx].get(self.conditioning_fallback_key) for idx in indices]
                    if not any(self._is_missing_entry(path) for path in fallback_paths):
                        record[self.conditioning_fallback_key] = {
                            "paths": fallback_paths,
                            "split_index": start,
                            "split_count": split_count,
                            "window": self.clip_length,
                        }
                        record[f"{self.conditioning_fallback_key}__split_index"] = start
                        record[f"{self.conditioning_fallback_key}__split_count"] = split_count

                records.append(record)

        if not records:
            raise ValueError("Empty Dataset")
        self.data = records
        self.size = len(self.data)

    def _cache_info(self, entry, row, key: str | None):
        if key is None:
            return None, 1
        return row.get(f"{key}__split_index"), row.get(f"{key}__split_count", 1)

    def _load_entry(self, entry, item_id):
        if isinstance(entry, dict) and isinstance(entry.get("paths"), list):
            frames = [
                load_image(str(Path(path)) if Path(path).is_absolute() else str(self.base_path / str(path)), item_id)
                for path in entry["paths"]
            ]
            images = [frame["Image"] for frame in frames]
            metadata = frames[0].get("Metadata") if frames else None
            return {"Image": np.stack(images, axis=0), "Metadata": metadata, "Id": item_id}
        return super()._load_entry(entry, item_id)

    def preprocess(self, payload: dict) -> np.ndarray:
        frames = payload["Image"] if isinstance(payload, dict) else payload
        frames = np.asarray(frames)
        if frames.ndim not in {3, 4}:
            raise ValueError(
                "VideoDataset expects stacked frames shaped as (T,H,W) or (T,H,W,C). "
                f"Got {tuple(frames.shape)}."
            )

        processed_frames: list[np.ndarray] = []
        for frame in frames:
            image = np.asarray(frame)
            if self.img_size is not None:
                image = resize(image, self.img_size, preserve_range=True)
            image = self.to_image(image)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            else:
                image = np.moveaxis(image, -1, 0)
            processed_frames.append(image.astype(self.img_datatype))

        clip = np.stack(processed_frames, axis=1)
        return clip.astype(self.img_datatype)

    def _resize_for_target_resolution(self, tensor: torch.Tensor, *, target_resolution: int | None) -> torch.Tensor:
        if target_resolution is None or not torch.is_tensor(tensor):
            return tensor
        if tensor.dim() != 4:
            return tensor
        channels, frames = tensor.shape[:2]
        if tensor.shape[-2:] == (int(target_resolution), int(target_resolution)):
            return tensor
        data = tensor.permute(1, 0, 2, 3).contiguous()
        data = F.interpolate(data, size=(int(target_resolution), int(target_resolution)), mode="bilinear", align_corners=False)
        return data.reshape(frames, channels, int(target_resolution), int(target_resolution)).permute(1, 0, 2, 3).contiguous()

    def _resolve_img_path(self, entry):
        if isinstance(entry, dict) and isinstance(entry.get("paths"), list) and entry["paths"]:
            paths = entry["paths"]
            return paths[len(paths) // 2]
        return super()._resolve_img_path(entry)


__all__ = ["VideoDataset"]
