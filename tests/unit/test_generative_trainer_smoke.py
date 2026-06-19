from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler

from training import (
    ConsistencyTrainer,
    DiffusionTrainer,
    EDMTrainer,
    FlowMatchingTrainer,
    ReflowTrainer,
    RectifiedFlowTrainer,
    TRAINER_REGISTRY,
    X0DenoisingTrainer,
)


def _make_scheduler_for_family(noise_family: str | None):
    family = str(noise_family or "ddpm").lower()
    if family in {"flow_matching", "rectified_flow", "reflow"}:
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    if family in {"x0_denoising", "consistency"}:
        return DDPMScheduler(num_train_timesteps=1000, prediction_type="sample")
    return DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")


class _DummyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        return x * self.weight

    def make_discriminator(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


class _TinyTextDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x, "text": f"prompt {idx}"}


class _TinyThreeDataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


def test_trainer_registry_contains_generative_keys() -> None:
    keys = set(TRAINER_REGISTRY.list())
    assert {"diffusion", "flow_matching", "consistency", "x0_denoising", "edm", "rectified_flow", "reflow"}.issubset(keys)
    assert TRAINER_REGISTRY.get("diffusion") is DiffusionTrainer
    assert TRAINER_REGISTRY.get("flow_matching") is FlowMatchingTrainer
    assert TRAINER_REGISTRY.get("consistency") is ConsistencyTrainer
    assert TRAINER_REGISTRY.get("x0_denoising") is X0DenoisingTrainer
    assert TRAINER_REGISTRY.get("edm") is EDMTrainer
    assert TRAINER_REGISTRY.get("rectified_flow") is RectifiedFlowTrainer
    assert TRAINER_REGISTRY.get("reflow") is ReflowTrainer


def test_generative_trainer_diffusion_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_diff"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "diff_last.pt").exists()
    assert (out / "diff_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_generative_trainer_flow_matching_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_flow"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "flow_matching",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = FlowMatchingTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "flow_last.pt").exists()
    assert (out / "flow_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_generative_trainer_diffusion_with_gan_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "gan_weight": 0.5,
            "gan_space": "prediction",
            "gan_start": 0,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_diff_gan"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
            "out_channels": 1,
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    csv_lines = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines
    assert "d_gan" in csv_lines[0]


def test_generative_trainer_freezes_discriminator_during_generator_backward(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "gan_weight": 0.5,
            "gan_space": "prediction",
            "gan_start": 0,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_diff_grad_freeze"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {"model_type": "diffusion", "scheduler": {}, "conditioning": "none", "out_channels": 1},
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer._setup(ds, val_dataset=ds, resume=None)
    trainer._discriminator_step = lambda **kwargs: 0.0  # type: ignore[method-assign]
    batch = next(iter(trainer.train_loader))
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.disc_optimizer.zero_grad(set_to_none=True)
    trainer._training_step(batch, epoch=1)
    assert all(param.grad is None for param in trainer.discriminator.parameters())
    assert trainer.disc_optimizer.defaults["betas"] == (0.5, 0.9)


def test_generative_trainer_gradient_accumulation_is_cross_batch(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_accum"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "gradient_accumulation_steps": 2,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyThreeDataset()
    trainer._setup(ds, val_dataset=ds, resume=None)
    calls = {"count": 0}
    original_step = trainer.optimizer.step

    def _count_step(*args, **kwargs):
        calls["count"] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = _count_step  # type: ignore[method-assign]
    trainer._train_epoch(epoch=1)
    assert calls["count"] == 2


def test_generative_trainer_text_conditioning_smoke(monkeypatch, tmp_path: Path) -> None:
    class _FakeTextAdapter:
        def __call__(self, model_input: torch.Tensor, cond, latent_norm=None):
            _ = latent_norm
            batch = len(cond)
            context = torch.ones(batch, 3, 4, device=model_input.device)
            return model_input, context

    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)
    monkeypatch.setattr("training.generative_trainer.build_text_conditioning_adapter", lambda **kwargs: _FakeTextAdapter())

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_text"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "conditioning": "text",
            "text_encoder": {"kind": "clip", "model_name": "fake/clip"},
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "text",
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyTextDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "diff_last.pt").exists()


def test_generative_trainer_consistency_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_consistency"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "consistency",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = ConsistencyTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "consistency_last.pt").exists()
    assert (out / "consistency_best.pt").exists()


def test_generative_trainer_edm_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_edm"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "edm",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = EDMTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "edm_last.pt").exists()
    assert (out / "edm_best.pt").exists()


def test_generative_trainer_rectified_flow_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_rectified"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "rectified_flow",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = RectifiedFlowTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "rectified_flow_last.pt").exists()
    assert (out / "rectified_flow_best.pt").exists()


def test_generative_trainer_reflow_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    pairs_dir = tmp_path / "reflow_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        torch.save({"z0": torch.randn(1, 8, 8), "z1": torch.randn(1, 8, 8)}, pairs_dir / f"{i:03d}.pt")

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_reflow"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "reflow_pairs_dir": str(pairs_dir),
        },
        "model": {
            "model_type": "reflow",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = ReflowTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "reflow_last.pt").exists()
    assert (out / "reflow_best.pt").exists()


def test_generative_trainer_sets_discriminator_eval_during_validation(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict, *, noise_family: str | None = None):
        return _make_scheduler_for_family(noise_family), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "gan_weight": 0.5,
            "gan_space": "prediction",
            "gan_start": 0,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_disc_eval"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
            "out_channels": 1,
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer._setup(ds, val_dataset=ds, resume=None)
    assert trainer.discriminator is not None
    sample = ds[0]
    batch = {"target": sample["target"].unsqueeze(0), "image": sample["image"].unsqueeze(0)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.discriminator.training is True
    trainer._run_step(batch, epoch=1, train=False)
    assert trainer.discriminator.training is False
