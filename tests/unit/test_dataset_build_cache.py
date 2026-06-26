from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.base import BaseDataset
from src.utils.dataset_utils import resolve_tensor_cache_subdir


def _write_split(root: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / "train.txt", sep="\t", index=False)


def test_build_cache_covers_target_conditioning_and_fallback_even_when_conditioning_disabled(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        [
            {"target": "t0.npy", "conditioning": "c0.npy", "fallback": "f0.npy"},
            {"target": "t1.npy", "conditioning": "", "fallback": "f1.npy"},
        ],
    )
    ds = BaseDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=False,
        target_key="target",
        conditioning_key="conditioning",
        conditioning_fallback_key="fallback",
        norm=False,
    )

    seen: list[str] = []

    def _fake_load_entry_tensor(row, item_id, key: str, preprocess: bool):
        _ = row, item_id, preprocess
        seen.append(key)
        return 0

    ds._load_entry_tensor = _fake_load_entry_tensor  # type: ignore[method-assign]
    total = ds.build_cache()

    assert total == 5
    assert seen == ["target", "conditioning", "fallback", "target", "fallback"]


def test_build_cache_restores_original_save_tensor_cache_flag(tmp_path: Path) -> None:
    _write_split(tmp_path, [{"target": "t0.npy"}])
    ds = BaseDataset(file_path=str(tmp_path), train=True, save_tensor_cache=False)

    def _fake_load_entry_tensor(row, item_id, key: str, preprocess: bool):
        _ = row, item_id, key, preprocess
        return 0

    ds._load_entry_tensor = _fake_load_entry_tensor  # type: ignore[method-assign]
    _ = ds.build_cache()
    assert ds.save_tensor_cache is False


def test_resolve_tensor_cache_subdir_is_semantic_for_2d_test_split() -> None:
    out = resolve_tensor_cache_subdir(
        {
            "tensor_cache_subdir": "cache",
            "img_size": 256,
            "window_size": 1,
        },
        dataset_class="datasets.ldct:LDCTDataset",
        train=False,
    )
    assert out == "cache/ldct/test/2d_256x256/ws1"


def test_resolve_tensor_cache_subdir_is_semantic_for_3d_train_split() -> None:
    out = resolve_tensor_cache_subdir(
        {
            "tensor_cache_subdir": "cache",
            "img_size": (64, 256, 256),
            "slice_count": 64,
        },
        dataset_class="datasets.medical3d:Medical3DDataset",
        train=True,
    )
    assert out == "cache/medical3d/train/3d_64x256x256/ws64"
