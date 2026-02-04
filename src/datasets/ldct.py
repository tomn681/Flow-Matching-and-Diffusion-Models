import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from utils.dataset_utils import absolute_path, maybe_unwrap, resolve_entry, split_volume_entry
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
    Lightweight integrity and failure tests for BaseDataset caching.
    """
    import tempfile

    if torch is None:
        raise RuntimeError("torch is required for dataset self-tests.")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = root / "data"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "sample.npy"
        np.save(sample_path, np.arange(6, dtype=np.float32).reshape(2, 3))
        (root / "train.txt").write_text("target\n" + "data/sample.npy\n")

        ds = BaseDataset(
            file_path=str(root),
            use_tensor_cache=True,
            save_tensor_cache=True,
            cache_subdir="cache",
        )
        first = ds[0]["target"].clone()
        cache_path = root / "cache" / "data" / "sample.pt"
        assert cache_path.exists(), "Cache file was not created."

        np.save(sample_path, np.zeros((2, 3), dtype=np.float32))
        second = ds[0]["target"]
        assert torch.equal(first, second), "Cache was not used on second access."

        ds_fail = BaseDataset(
            file_path=str(root),
            use_tensor_cache=False,
            save_tensor_cache=False,
            preprocess_kwargs={"bad_key": 1},
        )
        try:
            _ = ds_fail[0]
        except TypeError:
            pass
        else:
            raise AssertionError("Invalid preprocess kwargs did not raise TypeError.")
