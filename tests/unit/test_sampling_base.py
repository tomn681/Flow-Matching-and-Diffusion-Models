from __future__ import annotations

from pathlib import Path

import sampling.base as sampling_base
from sampling.base import BaseSampler


class _DummySampler(BaseSampler):
    def encode(self) -> None:
        return None

    def decode(self) -> None:
        return None

    def evaluate(self) -> None:
        return None


def test_build_tensor_cache_delegates(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def _fake_load_run_config(ckpt_dir: Path):
        called["ckpt_dir"] = ckpt_dir
        return {"training": {"save_tensor_cache": False}}

    def _fake_build_tensor_cache_from_config(**kwargs):
        called["kwargs"] = kwargs
        return 7

    monkeypatch.setattr(sampling_base, "load_run_config", _fake_load_run_config)
    monkeypatch.setattr(sampling_base, "build_tensor_cache_from_config", _fake_build_tensor_cache_from_config)

    sampler = _DummySampler(ckpt_dir=tmp_path, batch_size=5, seed=123, num_samples=9, save_tensor_cache=True)
    sampler.build_tensor_cache()

    assert called["ckpt_dir"] == tmp_path
    assert called["kwargs"]["batch_size"] == 5
    assert called["kwargs"]["seed"] == 123
    assert called["kwargs"]["num_samples"] == 9
    assert called["kwargs"]["desc"] == "build_tensor_cache"
    assert called["kwargs"]["evaluate"] is True
    assert called["kwargs"]["cfg"]["training"]["save_tensor_cache"] is True

