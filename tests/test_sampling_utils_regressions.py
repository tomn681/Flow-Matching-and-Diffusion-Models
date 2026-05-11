from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("torch")

from src.utils import sampling_utils as su


class _DummyDataset:
    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        return {"idx": idx}


def test_resolve_sample_indices_deterministic_subset():
    ds = _DummyDataset(20)
    a = su.resolve_sample_indices(ds, 7, seed=123)
    b = su.resolve_sample_indices(ds, 7, seed=123)
    c = su.resolve_sample_indices(ds, 7, seed=124)
    assert len(a) == 7
    assert a == b
    assert a != c
    assert len(set(a)) == 7


@pytest.mark.parametrize("num_samples", [None, 0, -1, 10, 11])
def test_resolve_sample_indices_full_dataset_cases(num_samples):
    ds = _DummyDataset(10)
    out = su.resolve_sample_indices(ds, num_samples, seed=42)
    assert out == list(range(10))


def test_resolve_checkpoint_prefers_best_then_last(tmp_path: Path):
    (tmp_path / "diff_last.pt").write_bytes(b"x")
    assert su.resolve_checkpoint(tmp_path, "diffusion").name == "diff_last.pt"
    (tmp_path / "diff_best.pt").write_bytes(b"x")
    assert su.resolve_checkpoint(tmp_path, "diffusion").name == "diff_best.pt"


def test_build_sampling_dataset_evaluate_switches_split_and_cache(monkeypatch):
    captured = {}

    def _fake_builder(training_cfg, model_cfg, train, cfg_path):
        captured["training_cfg"] = dict(training_cfg)
        captured["model_cfg"] = dict(model_cfg)
        captured["train"] = train
        captured["cfg_path"] = cfg_path
        return object()

    monkeypatch.setattr(su, "build_dataset_from_config", _fake_builder)
    cfg = {
        "training": {"split_file": "/tmp/train.txt", "tensor_cache_subdir": "cache"},
        "model": {"model_type": "diffusion"},
        "__config_path__": "/tmp/run/train_config.json",
    }

    su.build_sampling_dataset(cfg, data_txt=None, evaluate=True)
    tcfg = captured["training_cfg"]
    assert "split_file" not in tcfg
    assert tcfg["tensor_cache_subdir"] == "cache_eval"
    assert captured["train"] is False


def test_progress_batches_yields_expected_batches():
    ds = _DummyDataset(5)
    out = list(su.progress_batches(ds, batch_size=2, desc="test"))
    assert [idx for idx, _ in out] == [[0, 1], [2, 3], [4]]
    assert [len(samples) for _, samples in out] == [2, 2, 1]

