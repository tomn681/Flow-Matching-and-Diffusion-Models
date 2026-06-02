from __future__ import annotations

from utils import dataset_utils as du
from pathlib import Path


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


def test_cache_path_for_absolute_paths_outside_base_is_collision_safe():
    base = Path("/dataset/root")
    cache = Path("/dataset/root/cache")
    a = "/home2/LDCT/caseA/1-001.dcm"
    b = "/home2/LDCT/caseB/1-001.dcm"
    pa = du.cache_path_for_entry(base, cache, a)
    pb = du.cache_path_for_entry(base, cache, b)
    assert pa is not None and pb is not None
    assert pa != pb


def test_infer_dataset_class_supports_medical3d():
    resolved = du._infer_dataset_class({"dataset": "medical3d"})
    assert resolved == "datasets.medical3d:Medical3DDataset"
