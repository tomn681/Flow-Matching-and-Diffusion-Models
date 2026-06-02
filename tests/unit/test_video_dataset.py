from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.video import VideoDataset
from utils.dataset_utils import cache_path_for_entry


def _write_frame(path: Path, value: float) -> None:
    array = np.full((8, 8), value, dtype=np.float32)
    np.save(path, array)


def test_video_dataset_builds_temporal_clips_and_shapes(tmp_path: Path) -> None:
    rows = ["video_id\ttarget\tconditioning"]
    for idx in range(5):
        target = tmp_path / f"target_{idx:03d}.npy"
        cond = tmp_path / f"cond_{idx:03d}.npy"
        _write_frame(target, float(idx))
        _write_frame(cond, float(idx + 10))
        rows.append(f"vidA\t{target.name}\t{cond.name}")

    (tmp_path / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    ds = VideoDataset(
        file_path=str(tmp_path),
        train=True,
        clip_length=3,
        frame_stride=1,
        conditioning=True,
        img_size=8,
        norm=False,
    )
    assert len(ds) == 3
    sample = ds[0]
    assert tuple(sample["target"].shape) == (1, 3, 8, 8)
    assert tuple(sample["image"].shape) == (1, 3, 8, 8)
    assert sample["img_path"].endswith("target_001.npy")


def test_video_dataset_respects_frame_stride_within_clip(tmp_path: Path) -> None:
    rows = ["video_id\ttarget"]
    for idx in range(5):
        target = tmp_path / f"frame_{idx:03d}.npy"
        _write_frame(target, idx / 4.0)
        rows.append(f"vidA\t{target.name}")

    text = "\n".join(rows) + "\n"
    (tmp_path / "train.txt").write_text(text, encoding="utf-8")
    (tmp_path / "test.txt").write_text(text, encoding="utf-8")

    ds = VideoDataset(file_path=str(tmp_path), train=True, clip_length=3, frame_stride=2, img_size=8, norm=False)
    assert len(ds) == 1
    sample = ds[0]
    values = sample["target"][:, :, 0, 0].flatten().tolist()
    assert values == [0.0, 0.5, 1.0]
    assert sample["target"].shape[1] == 3


def test_video_dataset_target_resolution_resizes_spatial_dims_only(tmp_path: Path) -> None:
    rows = ["video_id\ttarget"]
    for idx in range(4):
        target = tmp_path / f"frame_{idx:03d}.npy"
        _write_frame(target, float(idx))
        rows.append(f"vidA\t{target.name}")

    text = "\n".join(rows) + "\n"
    (tmp_path / "train.txt").write_text(text, encoding="utf-8")
    (tmp_path / "test.txt").write_text(text, encoding="utf-8")

    ds = VideoDataset(file_path=str(tmp_path), train=True, clip_length=2, frame_stride=1, img_size=8, norm=False)
    sample = ds.__getitem__(0, target_resolution=4)
    assert tuple(sample["target"].shape) == (1, 2, 4, 4)


def test_video_dataset_cache_paths_are_unique_per_clip(tmp_path: Path) -> None:
    rows = ["video_id\ttarget"]
    for idx in range(4):
        target = tmp_path / f"frame_{idx:03d}.npy"
        _write_frame(target, float(idx))
        rows.append(f"vidA\t{target.name}")

    text = "\n".join(rows) + "\n"
    (tmp_path / "train.txt").write_text(text, encoding="utf-8")
    (tmp_path / "test.txt").write_text(text, encoding="utf-8")

    ds = VideoDataset(file_path=str(tmp_path), train=True, clip_length=2, frame_stride=1, img_size=8, norm=False)
    row0 = ds.data[0]
    row1 = ds.data[1]
    cp0 = cache_path_for_entry(ds.base_path, ds.cache_root, row0[ds.target_key], *ds._cache_info(row0[ds.target_key], row0, ds.target_key))
    cp1 = cache_path_for_entry(ds.base_path, ds.cache_root, row1[ds.target_key], *ds._cache_info(row1[ds.target_key], row1, ds.target_key))
    assert cp0 is not None and cp1 is not None
    assert cp0 != cp1
