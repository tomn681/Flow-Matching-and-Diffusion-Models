from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datasets.ldct import LDCTDataset


def _write_split(root: Path) -> None:
    rows = [{"Case": "C001", "SDCT": str(root / "sdct.npy"), "LDCT": str(root / "ldct.npy")}]
    pd.DataFrame(rows).to_csv(root / "train.txt", sep="\t", index=False, header=False)


def test_ldct_dataset_supports_conditioning_kwarg(tmp_path: Path) -> None:
    np.save(tmp_path / "sdct.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(tmp_path / "ldct.npy", np.ones((8, 8), dtype=np.float32))
    _write_split(tmp_path)
    ds = LDCTDataset(file_path=str(tmp_path), train=True, conditioning=True, img_size=8)
    assert ds.conditioning is True


def test_ldct_dataset_load_ldct_alias_overrides_conditioning(tmp_path: Path) -> None:
    np.save(tmp_path / "sdct.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(tmp_path / "ldct.npy", np.ones((8, 8), dtype=np.float32))
    _write_split(tmp_path)
    ds = LDCTDataset(file_path=str(tmp_path), train=True, conditioning=False, load_ldct=True, img_size=8)
    assert ds.conditioning is True

