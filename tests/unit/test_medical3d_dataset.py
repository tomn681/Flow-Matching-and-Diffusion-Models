from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from datasets.medical3d import Medical3DDataset
from utils.dataset_utils import cache_path_for_entry


def _write_split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "id\ttarget\tconditioning\n" + "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_medical3d_dataset_returns_channel_first_volume(tmp_path: Path):
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    np.save(tmp_path / "target.npy", volume)
    _write_split(tmp_path / "train.txt", [("case0", "target.npy", "target.npy")])
    _write_split(tmp_path / "test.txt", [("case0", "target.npy", "target.npy")])

    dataset = Medical3DDataset(file_path=str(tmp_path), train=True, conditioning=False, volume_size=(4, 5, 6))
    sample = dataset[0]

    assert sample["target"].shape == (1, 4, 5, 6)
    assert sample["image"].shape == (1, 4, 5, 6)
    assert sample["img_size"] == (4, 5, 6)


def test_medical3d_dataset_resizes_target_and_conditioning(tmp_path: Path):
    target = np.random.rand(8, 10, 12).astype(np.float32)
    cond = np.random.rand(8, 10, 12).astype(np.float32)
    np.save(tmp_path / "target.npy", target)
    np.save(tmp_path / "cond.npy", cond)
    _write_split(tmp_path / "train.txt", [("case0", "target.npy", "cond.npy")])
    _write_split(tmp_path / "test.txt", [("case0", "target.npy", "cond.npy")])

    dataset = Medical3DDataset(file_path=str(tmp_path), train=True, conditioning=True, volume_size=(8, 10, 12))
    sample = dataset.__getitem__(0, target_resolution=16)

    assert sample["target"].shape == (1, 16, 16, 16)
    assert sample["image"].shape == (1, 16, 16, 16)
    assert sample["img_size"] == (16, 16, 16)


def test_medical3d_dataset_build_cache_saves_volume_tensor(tmp_path: Path):
    volume = np.random.rand(6, 7, 8).astype(np.float32)
    np.save(tmp_path / "target.npy", volume)
    _write_split(tmp_path / "train.txt", [("case0", "target.npy", "target.npy")])
    _write_split(tmp_path / "test.txt", [("case0", "target.npy", "target.npy")])

    dataset = Medical3DDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=False,
        volume_size=(6, 7, 8),
        use_tensor_cache=True,
        save_tensor_cache=False,
    )
    cached = dataset.build_cache()

    cache_path = cache_path_for_entry(dataset.base_path, dataset.cache_root, "target.npy")
    assert cached == 2  # target + conditioning key both point to the same entry in the split row
    assert cache_path is not None and cache_path.exists()
    tensor = torch.load(cache_path)
    assert tuple(tensor.shape) == (1, 6, 7, 8)


def test_medical3d_dataset_accepts_channel_last_volume(tmp_path: Path):
    volume = np.random.rand(6, 7, 8, 2).astype(np.float32)
    np.save(tmp_path / "target.npy", volume)
    _write_split(tmp_path / "train.txt", [("case0", "target.npy", "target.npy")])
    _write_split(tmp_path / "test.txt", [("case0", "target.npy", "target.npy")])

    dataset = Medical3DDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=False,
        volume_size=(6, 7, 8),
        channel_order="channel_last",
    )
    sample = dataset[0]

    assert sample["target"].shape == (2, 6, 7, 8)


def test_medical3d_dataset_accepts_five_channel_channel_first_volume(tmp_path: Path):
    volume = np.random.rand(5, 6, 7, 8).astype(np.float32)
    np.save(tmp_path / "target.npy", volume)
    _write_split(tmp_path / "train.txt", [("case0", "target.npy", "target.npy")])
    _write_split(tmp_path / "test.txt", [("case0", "target.npy", "target.npy")])

    dataset = Medical3DDataset(file_path=str(tmp_path), train=True, conditioning=False, volume_size=(6, 7, 8))
    sample = dataset[0]

    assert sample["target"].shape == (5, 6, 7, 8)
