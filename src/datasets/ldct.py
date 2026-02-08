import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from skimage.transform import resize

from utils.dataset_utils import absolute_path, cache_path_for_entry, maybe_unwrap, resolve_entry, save_tensor_cache, split_volume_entry
from utils.utils import lot_id

from .base import BaseDataset


class LDCTDataset(BaseDataset):
    """
    LDCT/SDCT dataset. Overrides preprocess to apply HU conversion and CT normalization.
    """
    def __init__(
        self,
        file_path: str,
        train: bool = True,
        img_size: int | Tuple[int, int] | Tuple[int, int, int] | None = None,
        window_size: int = 1,
        norm: bool = True,
        img_datatype=np.float32,
        transforms=None,
        load_ldct: bool = False,
        names: Tuple[str, ...] = ("Case", "SDCT", "LDCT"),
        split_file: str | Path | None = None,
        use_tensor_cache: bool = True,
        save_tensor_cache: bool = False,
        cache_subdir: str = "cache",
    ):
        super().__init__(
            file_path=file_path,
            train=train,
            img_size=img_size,
            norm=norm,
            img_datatype=img_datatype,
            transforms=transforms,
            conditioning=load_ldct,
            id_key="Case",
            target_key=names[1],
            conditioning_key=names[2],
            split_names=names,
            split_file=split_file,
            use_tensor_cache=use_tensor_cache,
            save_tensor_cache=save_tensor_cache,
            cache_subdir=cache_subdir,
        )
        self.names = names
        self.window_size = int(window_size) if window_size is not None else 1
        self._build_ldct_index(names)

    def _build_ldct_index(self, names: Tuple[str, ...]) -> None:
        df = self._read_split_file(self.data_root, names=names)
        df = df.dropna().reset_index(drop=True)
        records = []
        for _, row in df.iterrows():
            sdct_path = absolute_path(self.data_root, row[names[1]])
            ldct_path = absolute_path(self.data_root, row[names[2]])
            sdct_opts = resolve_entry(self.data_root, row[names[1]], self.window_size) if sdct_path.is_dir() else split_volume_entry(str(sdct_path), self.window_size)
            ldct_opts = resolve_entry(self.data_root, row[names[2]], self.window_size) if ldct_path.is_dir() else split_volume_entry(str(ldct_path), self.window_size)
            if len(sdct_opts) != len(ldct_opts):
                logging.warning(
                    "Skipping case %s due to mismatched slice counts (SDCT=%d, LDCT=%d)",
                    row["Case"], len(sdct_opts), len(ldct_opts)
                )
                continue
            for sdct_idx, (sdct_paths, ldct_paths) in enumerate(zip(sdct_opts, ldct_opts)):
                sdct_entry = maybe_unwrap(sdct_paths) if isinstance(sdct_paths, (list, tuple)) else sdct_paths
                ldct_entry = maybe_unwrap(ldct_paths) if isinstance(ldct_paths, (list, tuple)) else ldct_paths
                sdct_split_idx = sdct_entry.get("split_index") if isinstance(sdct_entry, dict) else sdct_idx
                sdct_split_cnt = sdct_entry.get("split_count", len(sdct_opts)) if isinstance(sdct_entry, dict) else len(sdct_opts)
                ldct_split_idx = ldct_entry.get("split_index") if isinstance(ldct_entry, dict) else sdct_idx
                ldct_split_cnt = ldct_entry.get("split_count", len(ldct_opts)) if isinstance(ldct_entry, dict) else len(ldct_opts)
                records.append({
                    "Case": row["Case"],
                    names[1]: sdct_entry,
                    names[2]: ldct_entry,
                    f"{names[1]}__split_index": sdct_split_idx,
                    f"{names[1]}__split_count": sdct_split_cnt,
                    f"{names[2]}__split_index": ldct_split_idx,
                    f"{names[2]}__split_count": ldct_split_cnt,
                })
        if not records:
            raise ValueError("Empty Dataset")
        df = pd.DataFrame(records)
        df = lot_id(df, "Case", names[1])
        self.data = df.to_dict("records")
        self.size = len(self.data)

    def _cache_info(self, entry, row, key: str | None):
        if key is None:
            return None, 1
        return row.get(f"{key}__split_index"), row.get(f"{key}__split_count", 1)


    def preprocess(
        self,
        payload: dict,
        MIN_B: float = -1024,
        MAX_B: float = 3072,
        slope: float = 1.0,
        intersept: float = -1024,
    ) -> np.ndarray:
        img = payload["Image"] if isinstance(payload, dict) else payload
        meta = payload.get("Metadata") if isinstance(payload, dict) else None
        if meta is not None:
            try:
                slope = float(meta.get("Rescale Slope", slope))
                intersept = float(meta.get("Rescale Intercept", intersept))
            except (TypeError, ValueError):
                pass
        img = np.asarray(img) * slope + intersept
        if self.img_size is not None:
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
                img = resize(img, self.img_size, preserve_range=True)
                img = np.transpose(img, (2, 0, 1))
            else:
                img = resize(img, self.img_size, preserve_range=True)
        if self.norm:
            img = (img - MIN_B) / (MAX_B - MIN_B)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return img.astype(self.img_datatype)


class LDCTAttentionDataset(LDCTDataset):
    """
    LDCT dataset that skips preprocessing for conditioning inputs (e.g., VAE latents).
    """
    def __getitem__(self, idx):
        """
        getitem Method

        Loads a single sample, applying preprocess to targets only.

        Inputs:
            - idx: (Int) Sample index.

        Outputs:
            - target: (dict) Keys: image, target, img_id, img_path, img_size
        """
        row = self.data[idx]
        target_key = self.target_key
        cond_key = self.conditioning_key

        item_id = row.get(self.id_key) if self.id_key else None
        tgt_entry = row[target_key]
        tgt_split_index, tgt_split_count = self._cache_info(tgt_entry, row, target_key)
        tgt_cache_path = cache_path_for_entry(
            self.base_path,
            self.cache_root,
            tgt_entry,
            tgt_split_index,
            tgt_split_count,
        )
        if self.use_tensor_cache and tgt_cache_path is not None and tgt_cache_path.exists():
            tgt = torch.load(tgt_cache_path)
            tgt = torch.as_tensor(tgt).float().contiguous()
        else:
            tgt_payload = self._load_entry(tgt_entry, item_id)
            try:
                tgt = self.preprocess(tgt_payload, **self.preprocess_kwargs) if self.preprocess_kwargs else self.preprocess(tgt_payload)
            except TypeError as exc:
                raise TypeError(f"Invalid preprocess kwargs for {self.__class__.__name__}: {self.preprocess_kwargs}") from exc
            tgt = torch.as_tensor(tgt).float().contiguous()
            if self.save_tensor_cache and tgt_cache_path is not None and not tgt_cache_path.exists():
                save_tensor_cache(tgt, tgt_cache_path)

        img = None
        if self.conditioning:
            if cond_key is None:
                raise KeyError("Conditioning requested but no conditioning column provided.")
            cond_entry = row[cond_key]
            cond_split_index, cond_split_count = self._cache_info(cond_entry, row, cond_key)
            cond_cache_path = cache_path_for_entry(
                self.base_path,
                self.cache_root,
                cond_entry,
                cond_split_index,
                cond_split_count,
            )
            if self.use_tensor_cache and cond_cache_path is not None and cond_cache_path.exists():
                img = torch.load(cond_cache_path)
                img = torch.as_tensor(img).float().contiguous()
            else:
                cond_payload = self._load_entry(cond_entry, item_id)
                img = cond_payload.get("Image") if isinstance(cond_payload, dict) else cond_payload
                img = torch.as_tensor(img).float().contiguous()
                if self.save_tensor_cache and cond_cache_path is not None and not cond_cache_path.exists():
                    save_tensor_cache(img, cond_cache_path)

        if self.transforms is not None:
            if self.train and not self.conditioning:
                tgt = self.transforms(tgt)
            else:
                img, tgt = self.transforms(img, tgt)

        if img is None:
            img = tgt

        target = {
            "image": img,
            "target": tgt,
            "img_id": item_id,
            "img_path": self._resolve_img_path(row.get(target_key)),
            "img_size": self.img_size,
        }
        return target


def build_ldct_from_config(training_cfg: dict, _model_cfg: dict | None, train: bool):
    """
    Factory for LDCTDataset used by the dataset registry.
    """
    data_root = Path(training_cfg["data_root"])
    img_size = training_cfg.get("img_size")
    window_size = training_cfg.get("window_size", training_cfg.get("slice_count", 1))
    load_ldct = bool(training_cfg.get("load_ldct", False))
    use_tensor_cache = bool(training_cfg.get("use_tensor_cache", True))
    save_tensor_cache = bool(training_cfg.get("save_tensor_cache", False))
    cache_subdir = training_cfg.get("tensor_cache_subdir", "cache")
    norm = training_cfg.get("norm", True)
    return LDCTDataset(
        str(data_root),
        train=train,
        img_size=img_size,
        window_size=window_size,
        norm=norm,
        load_ldct=load_ldct,
        use_tensor_cache=use_tensor_cache,
        save_tensor_cache=save_tensor_cache,
        cache_subdir=cache_subdir,
    )


def run_self_tests() -> None:
    """
    Lightweight tests for LDCTDataset slicing and preprocessing.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sdct_path = data_dir / "sdct.npy"
        ldct_path = data_dir / "ldct.npy"
        volume = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        np.save(sdct_path, volume)
        np.save(ldct_path, volume)
        (root / "train.txt").write_text("Case\tSDCT\tLDCT\nC1\tdata/sdct.npy\tdata/ldct.npy\n")

        ds = LDCTDataset(
            file_path=str(root),
            window_size=1,
            img_size=None,
            load_ldct=True,
        )
        assert len(ds) == 3, "LDCTDataset should expand each slice for window_size=1."
        sample = ds[0]
        assert sample["target"].shape[0] == 1, "LDCTDataset should add channel dimension."
        assert sample["image"] is not None, "Conditioning image should be present when load_ldct=True."
