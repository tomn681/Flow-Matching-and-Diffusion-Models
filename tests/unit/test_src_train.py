from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path


def test_src_train_dispatches_via_registry_and_applies_overrides(monkeypatch, tmp_path: Path) -> None:
    src_train = importlib.import_module("src.train")
    calls: dict[str, object] = {}

    class _DummyTrainer:
        @classmethod
        def from_config(cls, cfg):
            calls["cfg"] = cfg
            return cls()

        def fit(self, dataset, val_dataset=None):
            calls["fit"] = (dataset, val_dataset)

    def _fake_build_train_val_datasets(cfg):
        calls["dataset_cfg"] = cfg
        return ["train"], ["val"]

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}")

    monkeypatch.setattr(
        src_train,
        "load_json_config",
        lambda path: {
            "model": {"model_type": "diffusion", "resolution": 64},
            "training": {"epochs": 1, "batch_size": 2, "data_root": "original"},
            "dataset": {"class": "datasets.ldct:LDCTDataset"},
            "__config_path__": str(path),
        },
    )
    monkeypatch.setattr(src_train, "build_train_val_datasets", _fake_build_train_val_datasets)
    monkeypatch.setattr(src_train.TRAINER_REGISTRY, "get", lambda key: _DummyTrainer if key == "vae" else None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "--trainer",
            "vae",
            "--config",
            str(cfg_path),
            "--data-root",
            str(tmp_path / "data"),
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--img-size",
            "128",
        ],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src_train.main()

    dep_warnings = [item for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(dep_warnings) >= 2
    assert any("python -m src.train" in str(item.message) for item in dep_warnings)
    assert any("overrides config model_type" in str(item.message) for item in dep_warnings)

    dataset_cfg = calls["dataset_cfg"]
    assert dataset_cfg["training"]["data_root"] == str(tmp_path / "data")
    assert dataset_cfg["training"]["epochs"] == 5
    assert dataset_cfg["training"]["batch_size"] == 8
    assert dataset_cfg["training"]["img_size"] == 128
    assert dataset_cfg["model"]["resolution"] == 128
    assert dataset_cfg["model"]["model_type"] == "vae"
    assert calls["cfg"]["model"]["model_type"] == "vae"
    assert calls["fit"] == (["train"], ["val"])
