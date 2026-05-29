from __future__ import annotations

from utils import dataset_utils as du


def test_build_train_val_datasets_infers_img_size_from_model_resolution(monkeypatch):
    captured = []

    def _fake_build(training_cfg, model_cfg, train, cfg_path=None, dataset_cfg=None):
        captured.append((dict(training_cfg), dict(model_cfg), train, cfg_path, dict(dataset_cfg or {})))
        return object()

    monkeypatch.setattr(du, "build_dataset_from_config", _fake_build)

    cfg = {
        "training": {"batch_size": 2},
        "model": {"model_type": "vae", "resolution": 256},
        "dataset": {"class": "datasets.base:BaseDataset"},
    }

    du.build_train_val_datasets(cfg)

    assert len(captured) == 2
    train_call, val_call = captured
    assert train_call[0]["img_size"] == 256
    assert val_call[0]["img_size"] == 256
    assert train_call[2] is True
    assert val_call[2] is False
