from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


_ROOT_TRAIN = Path(__file__).resolve().parents[2] / "train.py"
_SPEC = importlib.util.spec_from_file_location("root_train_entry", _ROOT_TRAIN)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load train entry from {_ROOT_TRAIN}.")
train_entry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_entry)


class _TinyDataset:
    def __init__(self, with_image: bool) -> None:
        self.with_image = with_image

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> dict:
        t = torch.full((1, 4, 4), float(idx))
        out = {"target": t}
        if self.with_image:
            out["image"] = t + 1.0
        return out


class _DummyPosterior:
    def __init__(self, x: torch.Tensor) -> None:
        self._x = x

    def mode(self) -> torch.Tensor:
        return self._x


class _DummyVAE(nn.Module):
    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor, normalize: bool = False):
        if normalize:
            return x * 0.25
        return _DummyPosterior(x * 0.25)


def test_encode_latents_mode_writes_train_and_val_cache(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {"batch_size": 2, "num_workers": 0, "manual_device": "cpu"},
        "model": {
            "model_type": "latent_diffusion",
            "latent_cache_dir": str(tmp_path / "latents"),
            "vae_checkpoint": str(tmp_path / "dummy_vae.pt"),
            "vae": {"latent_type": "kl"},
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}")

    monkeypatch.setattr(train_entry, "load_json_config", lambda _: cfg)
    monkeypatch.setattr(
        train_entry,
        "build_train_val_datasets",
        lambda _cfg: (_TinyDataset(with_image=True), _TinyDataset(with_image=True)),
    )
    monkeypatch.setattr(train_entry, "_load_frozen_vae_from_cfg", lambda _cfg, _device: _DummyVAE())

    train_entry.encode_latents_from_config(cfg_path)

    train_files = sorted((tmp_path / "latents" / "train").glob("*.pt"))
    val_files = sorted((tmp_path / "latents" / "val").glob("*.pt"))
    assert len(train_files) == 3
    assert len(val_files) == 3

    sample = torch.load(train_files[0], map_location="cpu")
    assert set(sample.keys()) == {"target", "image"}
    assert tuple(sample["target"].shape) == (1, 4, 4)
    assert tuple(sample["image"].shape) == (1, 4, 4)


def test_dispatch_train_uses_latent_cache_dataset_when_presaved_for_latent_rf(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {"batch_size": 2, "num_workers": 0, "manual_device": "cpu"},
        "model": {
            "model_type": "latent_rectified_flow",
            "use_presaved_latents": True,
            "latent_cache_dir": str(tmp_path / "latents"),
        },
    }
    latents_train = tmp_path / "latents" / "train"
    latents_val = tmp_path / "latents" / "val"
    latents_train.mkdir(parents=True)
    latents_val.mkdir(parents=True)
    torch.save({"target": torch.zeros(1, 4, 4)}, latents_train / "000.pt")
    torch.save({"target": torch.zeros(1, 4, 4)}, latents_val / "000.pt")

    called = {}

    def _fake_trainer(train_ds, cfg_path, val_dataset=None, resume=None):
        called["train_len"] = len(train_ds)
        called["val_len"] = len(val_dataset)

    monkeypatch.setattr(train_entry, "load_json_config", lambda _: cfg)
    monkeypatch.setattr(train_entry, "build_train_val_datasets", lambda _cfg: (_TinyDataset(True), _TinyDataset(True)))
    monkeypatch.setitem(train_entry.TRAINERS, "latent_rectified_flow", _fake_trainer)

    train_entry.dispatch_train(tmp_path / "cfg.json", resume=None)
    assert called["train_len"] == 1
    assert called["val_len"] == 1


def test_dispatch_train_routes_phase_i_types_to_registry_trainers(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def _capture_registry(key, dataset, json_path, *, val_dataset=None, resume=None):
        _ = dataset, json_path, val_dataset, resume
        captured.append(key)

    monkeypatch.setattr(train_entry, "_train_via_registry", _capture_registry)
    monkeypatch.setattr(train_entry, "build_train_val_datasets", lambda _cfg: (_TinyDataset(True), _TinyDataset(True)))

    for model_type, expected_key in [
        ("diffusion", "diffusion"),
        ("flow_matching", "flow_matching"),
        ("latent_diffusion", "latent_diffusion"),
        ("latent_flow_matching", "latent_flow_matching"),
        ("consistency", "consistency"),
        ("edm", "edm"),
        ("rectified_flow", "rectified_flow"),
        ("reflow", "reflow"),
        ("distillation", "distillation"),
    ]:
        cfg = {"training": {}, "model": {"model_type": model_type}}
        monkeypatch.setattr(train_entry, "load_json_config", lambda _p, cfg=cfg: cfg)
        train_entry.dispatch_train(tmp_path / f"{model_type}.json", resume=None)
        assert captured[-1] == expected_key


def test_dispatch_train_forwards_scheduler_resume_mode_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_registry(key, dataset, json_path, *, val_dataset=None, resume=None, scheduler_resume_mode=None, overrides=None):
        _ = dataset, json_path, val_dataset, resume, overrides
        captured["key"] = key
        captured["scheduler_resume_mode"] = scheduler_resume_mode

    cfg = {"training": {}, "model": {"model_type": "diffusion"}}
    monkeypatch.setattr(train_entry, "load_json_config", lambda _p: cfg)
    monkeypatch.setattr(train_entry, "_train_via_registry", _capture_registry)
    monkeypatch.setattr(train_entry, "build_train_val_datasets", lambda _cfg: (_TinyDataset(True), _TinyDataset(True)))

    train_entry.dispatch_train(tmp_path / "diffusion.json", resume=None, scheduler_resume_mode="continue")

    assert captured["key"] == "diffusion"
    assert captured["scheduler_resume_mode"] == "continue"


def test_dispatch_train_set_scheduler_adds_horizon_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_registry(key, dataset, json_path, *, val_dataset=None, resume=None, scheduler_resume_mode=None, overrides=None):
        _ = dataset, json_path, val_dataset, resume, key
        captured["scheduler_resume_mode"] = scheduler_resume_mode
        captured["overrides"] = list(overrides or [])

    cfg = {"training": {"lr_scheduler": {"name": "warmup_cosine"}}, "model": {"model_type": "diffusion"}}
    monkeypatch.setattr(train_entry, "load_json_config", lambda _p, overrides=None: cfg)
    monkeypatch.setattr(train_entry, "_train_via_registry", _capture_registry)
    monkeypatch.setattr(train_entry, "build_train_val_datasets", lambda _cfg: (_TinyDataset(True), _TinyDataset(True)))

    train_entry.dispatch_train(tmp_path / "diffusion.json", resume=None, scheduler_resume_mode="continue", set_scheduler=100)

    assert captured["scheduler_resume_mode"] == "continue"
    assert "training.lr_scheduler.params.epochs=100" in captured["overrides"]
    assert "training.epochs=100" not in captured["overrides"]


def test_train_interrupt_label_is_mode_specific() -> None:
    assert train_entry._interrupt_label("train") == "Training"
    assert train_entry._interrupt_label("encode_latents") == "Latent encoding"
    assert train_entry._interrupt_label("train", debug_visual_only=True) == "Debug visual export"
