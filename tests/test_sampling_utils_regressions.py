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


def test_resolve_checkpoint_legacy_diffusers_safetensors(tmp_path: Path):
    unet_dir = tmp_path / "unet"
    unet_dir.mkdir(parents=True)
    legacy_ckpt = unet_dir / "diffusion_pytorch_model.safetensors"
    legacy_ckpt.write_bytes(b"x")
    assert su.resolve_checkpoint(tmp_path, "diffusion") == legacy_ckpt


def test_load_run_config_accepts_legacy_folder_without_model_index(tmp_path: Path):
    scheduler_dir = tmp_path / "scheduler"
    unet_dir = tmp_path / "unet"
    scheduler_dir.mkdir(parents=True)
    unet_dir.mkdir(parents=True)

    (scheduler_dir / "scheduler_config.json").write_text(
        '{"_class_name":"DDPMScheduler","num_train_timesteps":1000}'
    )
    (unet_dir / "config.json").write_text(
        '{"in_channels":2,"out_channels":1,"sample_size":256,"layers_per_block":2,'
        '"block_out_channels":[128,128,256,256,512,512],'
        '"down_block_types":["DownBlock2D"],"up_block_types":["UpBlock2D"]}'
    )

    cfg = su.load_run_config(tmp_path)
    assert cfg["model"]["model_type"] == "diffusion"
    assert cfg["model"]["conditioning"] == "concatenate"
    assert cfg["model"]["legacy_source"]["model_index"] is None
    assert cfg["__config_path__"].endswith("scheduler/scheduler_config.json")


def test_build_sampling_dataset_evaluate_switches_split_and_cache(monkeypatch):
    captured = {}

    def _fake_builder(training_cfg, model_cfg, train, cfg_path, dataset_cfg=None):
        captured["training_cfg"] = dict(training_cfg)
        captured["model_cfg"] = dict(model_cfg)
        captured["train"] = train
        captured["cfg_path"] = cfg_path
        captured["dataset_cfg"] = dict(dataset_cfg or {})
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
    assert tcfg["tensor_cache_subdir"] == "cache"
    assert captured["train"] is False
    assert captured["dataset_cfg"] == {}


def test_build_sampling_dataset_uses_semantic_test_cache_namespace(monkeypatch):
    captured = {}

    def _fake_builder(training_cfg, model_cfg, train, cfg_path, dataset_cfg=None):
        captured["training_cfg"] = dict(training_cfg)
        captured["train"] = train
        return object()

    monkeypatch.setattr(su, "build_dataset_from_config", _fake_builder)
    cfg = {
        "training": {
            "tensor_cache_subdir": "cache",
            "img_size": 256,
            "window_size": 3,
        },
        "model": {"model_type": "diffusion"},
        "dataset": {"class": "datasets.ldct:LDCTDataset"},
        "__config_path__": "/tmp/run/train_config.json",
    }

    su.build_sampling_dataset(cfg, data_txt=None, evaluate=True)
    assert captured["train"] is False
    assert captured["training_cfg"]["tensor_cache_subdir"] == "cache"


def test_progress_batches_yields_expected_batches():
    ds = _DummyDataset(5)
    out = list(su.progress_batches(ds, batch_size=2, desc="test"))
    assert [idx for idx, _ in out] == [[0, 1], [2, 3], [4]]
    assert [len(samples) for _, samples in out] == [2, 2, 1]
