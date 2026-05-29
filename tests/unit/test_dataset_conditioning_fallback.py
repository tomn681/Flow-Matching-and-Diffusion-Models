from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.base import BaseDataset


def _write_split(root: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(root / "train.txt", sep="\t", index=False)


def test_base_dataset_conditioning_fallback_uses_fallback_entry(tmp_path: Path) -> None:
    target = np.ones((8, 8), dtype=np.float32)
    cond = np.zeros((8, 8), dtype=np.float32)
    fallback = np.full((8, 8), 0.25, dtype=np.float32)

    np.save(tmp_path / "target.npy", target)
    np.save(tmp_path / "cond.npy", cond)
    np.save(tmp_path / "fallback.npy", fallback)

    _write_split(
        tmp_path,
        [
            {"target": str(tmp_path / "target.npy"), "conditioning": "", "fallback": str(tmp_path / "fallback.npy")},
            {"target": str(tmp_path / "target.npy"), "conditioning": str(tmp_path / "cond.npy"), "fallback": str(tmp_path / "fallback.npy")},
        ],
    )

    ds = BaseDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=True,
        target_key="target",
        conditioning_key="conditioning",
        conditioning_fallback_key="fallback",
        norm=False,
    )

    item0 = ds[0]
    item1 = ds[1]
    assert torch.equal(item0["image"], torch.from_numpy(fallback).float())
    assert torch.equal(item1["image"], torch.from_numpy(cond).float())


def test_base_dataset_conditioning_fallback_raises_when_both_missing(tmp_path: Path) -> None:
    target = np.ones((8, 8), dtype=np.float32)
    np.save(tmp_path / "target.npy", target)

    _write_split(
        tmp_path,
        [{"target": str(tmp_path / "target.npy"), "conditioning": "", "fallback": ""}],
    )

    ds = BaseDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=True,
        target_key="target",
        conditioning_key="conditioning",
        conditioning_fallback_key="fallback",
        norm=False,
    )

    with pytest.raises(KeyError):
        _ = ds[0]


def test_base_dataset_conditioning_fallback_recovers_from_missing_conditioning_file(tmp_path: Path) -> None:
    target = np.ones((8, 8), dtype=np.float32)
    fallback = np.full((8, 8), 0.75, dtype=np.float32)
    np.save(tmp_path / "target.npy", target)
    np.save(tmp_path / "fallback.npy", fallback)

    missing_cond = tmp_path / "missing_conditioning.npy"
    _write_split(
        tmp_path,
        [
            {
                "target": str(tmp_path / "target.npy"),
                "conditioning": str(missing_cond),
                "fallback": str(tmp_path / "fallback.npy"),
            }
        ],
    )

    ds = BaseDataset(
        file_path=str(tmp_path),
        train=True,
        conditioning=True,
        target_key="target",
        conditioning_key="conditioning",
        conditioning_fallback_key="fallback",
        norm=False,
    )

    item = ds[0]
    assert torch.equal(item["image"], torch.from_numpy(fallback).float())
