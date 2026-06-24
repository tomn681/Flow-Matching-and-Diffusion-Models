from __future__ import annotations

from pathlib import Path
import stat

import pytest
import torch

import utils


def test_save_checkpoint_writes_versioned_sidecar_and_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "model_last.pt"
    payload = {
        "model": {"weight": torch.ones(2, 2)},
        "optimizer": {"state": {}},
        "epoch": 3,
        "best_metric": 0.1,
    }

    utils.save_checkpoint(payload, path)

    assert path.exists()
    assert path.with_suffix(".safetensors").exists()

    loaded = utils.safe_torch_load(path, map_location="cpu")
    assert loaded["format_version"] == utils.CHECKPOINT_FORMAT_VERSION
    assert torch.equal(loaded["model"]["weight"], torch.ones(2, 2))
    assert loaded["weights_path"].endswith(".safetensors")


def test_safe_torch_load_migrates_legacy_epoch_keys(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model": {"w": torch.tensor(1.0)}, "current_epoch": 5}, path)

    loaded = utils.safe_torch_load(path, map_location="cpu")
    assert loaded["epoch"] == 5
    assert loaded["format_version"] == utils.CHECKPOINT_FORMAT_VERSION


def test_safe_torch_load_refuses_old_torch_without_weights_only(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "payload.pt"
    torch.save({"x": torch.tensor(1.0)}, path)

    real_load = torch.load

    def _fake_load(*args, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("weights_only unsupported")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _fake_load)
    with pytest.raises(RuntimeError, match="weights_only=True"):
        utils.safe_torch_load(path, map_location="cpu")


def test_save_checkpoint_normalizes_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "model_best.pt"
    payload = {
        "model": {"weight": torch.ones(2, 2)},
        "epoch": 1,
    }

    utils.save_checkpoint(payload, path)

    pt_mode = stat.S_IMODE(path.stat().st_mode)
    st_mode = stat.S_IMODE(path.with_suffix(".safetensors").stat().st_mode)

    assert pt_mode == 0o644
    assert st_mode == 0o644
