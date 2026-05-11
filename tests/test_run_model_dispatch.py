from __future__ import annotations

import pytest


pytest.importorskip("torch")

from src import run_model as rm


class _DummyHandler:
    init_kwargs = None
    called = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def encode(self):
        type(self).called = "encode"

    def decode(self):
        type(self).called = "decode"

    def evaluate(self):
        type(self).called = "evaluate"

    def sample(self):
        type(self).called = "sample"


def test_run_model_forwards_new_flags(monkeypatch, tmp_path):
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()

    def _fake_load_run_config(path):
        assert path == ckpt_dir
        return {"model": {"model_type": "diffusion"}, "training": {}}

    monkeypatch.setattr(rm, "load_run_config", _fake_load_run_config)
    monkeypatch.setattr(rm, "HANDLER_REGISTRY", {"diffusion": _DummyHandler})
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model.py",
            "--ckpt_dir",
            str(ckpt_dir),
            "--mode",
            "evaluate",
            "--batch_size",
            "64",
            "--num_samples",
            "128",
            "--save",
            "--save_input",
            "--save_conditioning",
        ],
    )

    rm.main()

    assert _DummyHandler.called == "evaluate"
    assert _DummyHandler.init_kwargs["batch_size"] == 64
    assert _DummyHandler.init_kwargs["num_samples"] == 128
    assert _DummyHandler.init_kwargs["save"] is True
    assert _DummyHandler.init_kwargs["save_input"] is True
    assert _DummyHandler.init_kwargs["save_conditioning"] is True

