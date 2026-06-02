from __future__ import annotations

from pathlib import Path

import torch

from datasets.base import BaseDataset
from utils.dataset_utils import cache_path_for_entry


def _make_dataset(tmp_path: Path, *, conditioning: bool = True) -> BaseDataset:
    (tmp_path / "train.txt").write_text("id\ttarget\tconditioning\nx\ta.npy\tb.npy\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("id\ttarget\tconditioning\nx\ta.npy\tb.npy\n", encoding="utf-8")
    ds = BaseDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=conditioning,
        id_key="id",
        target_key="target",
        conditioning_key="conditioning",
        use_tensor_cache=False,
        save_tensor_cache=False,
    )
    return ds


def test_dataset_resize_resizes_target_and_conditioning_together(tmp_path: Path, monkeypatch) -> None:
    ds = _make_dataset(tmp_path)

    def _fake_load_entry_tensor(row, item_id, key: str, preprocess: bool):
        _ = row, item_id, preprocess
        return torch.zeros(1, 16, 16) if key == "target" else torch.ones(1, 16, 16)

    monkeypatch.setattr(ds, "_load_entry_tensor", _fake_load_entry_tensor)
    sample = ds.__getitem__(0, target_resolution=8)
    assert tuple(sample["target"].shape) == (1, 8, 8)
    assert tuple(sample["image"].shape) == (1, 8, 8)


def test_dataset_resize_none_is_identity(tmp_path: Path, monkeypatch) -> None:
    ds = _make_dataset(tmp_path, conditioning=False)

    def _fake_load_target_tensor(row, item_id):
        _ = row, item_id
        return torch.randn(1, 12, 12)

    monkeypatch.setattr(ds, "_load_target_tensor", _fake_load_target_tensor)
    sample = ds.__getitem__(0, target_resolution=None)
    assert tuple(sample["target"].shape) == (1, 12, 12)
    assert tuple(sample["image"].shape) == (1, 12, 12)


def test_dataset_resize_does_not_mutate_cache(tmp_path: Path, monkeypatch) -> None:
    ds = _make_dataset(tmp_path, conditioning=False)
    ds.use_tensor_cache = True
    entry = ds.data[0][ds.target_key]
    cp = cache_path_for_entry(ds.base_path, ds.cache_root, entry, None, 1)
    assert cp is not None
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(1, 16, 16), cp)
    before = cp.stat().st_mtime_ns

    sample = ds.__getitem__(0, target_resolution=8)
    assert tuple(sample["target"].shape) == (1, 8, 8)
    after = cp.stat().st_mtime_ns
    assert before == after

