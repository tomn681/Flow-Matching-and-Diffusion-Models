from __future__ import annotations

from pathlib import Path

import torch

from core.types import NoisyBatch
from models.factory import ModelFactory
from training import TRAINER_REGISTRY
from training.controlnet_trainer import ControlNetTrainer


class _FakeBaseUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_only = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t, context_ca=None, controlnet_residuals=None):
        _ = t, context_ca
        out = x * self.base_only
        if controlnet_residuals is not None:
            out = out + controlnet_residuals["mid_residual"]
        return out


class _FakeControlNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.control_only = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, controlnet_cond, encoder_hidden_states=None):
        _ = x, t, encoder_hidden_states
        mid = controlnet_cond * self.control_only
        return {
            "down_residuals": [],
            "mid_residual": mid,
        }


class _FakeDDPMNoise:
    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noisy = clean + 0.25
        target = torch.zeros_like(clean, device=device)
        timesteps = torch.zeros(clean.size(0), device=device, dtype=torch.long)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)


def test_controlnet_trainer_in_registry() -> None:
    assert TRAINER_REGISTRY.get("controlnet") is ControlNetTrainer


def test_model_factory_builds_controlnet() -> None:
    cfg = {
        "model": {
            "model_type": "controlnet",
            "controlnet": {
                "in_channels": 4,
                "conditioning_channels": 3,
                "layers_per_block": 1,
                "block_out_channels": [32, 64, 64, 64],
                "cross_attention_dim": 16,
                "attention_head_dim": 8,
            },
        }
    }
    model = ModelFactory.build(cfg)
    assert model.__class__.__name__ == "ControlNetND"


def test_controlnet_trainer_base_unet_frozen_and_controlnet_gets_grad(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "output_dir": str(tmp_path / "controlnet"),
            "validate": False,
            "use_amp": False,
            "scheduler": "ddpm",
            "num_train_timesteps": 10,
        },
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": "unused",
            "scheduler": {"name": "ddpm", "num_train_timesteps": 10},
        },
    }
    train_ds = [{"target": torch.ones(1, 4, 4), "image": torch.ones(1, 4, 4)}]
    base_unet = _FakeBaseUNet()
    controlnet = _FakeControlNet()
    trainer = ControlNetTrainer(
        config=cfg,
        model_override=controlnet,
        base_unet_override=base_unet,
        noise_override=_FakeDDPMNoise(),
    )
    trainer.fit(train_ds, val_dataset=None)

    assert all(not p.requires_grad for p in base_unet.parameters())
    assert base_unet.base_only.grad is None
    assert controlnet.control_only.grad is not None


def test_controlnet_checkpoint_does_not_include_base_unet_state(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "output_dir": str(tmp_path / "controlnet"),
            "validate": False,
            "use_amp": False,
            "scheduler": "ddpm",
            "num_train_timesteps": 10,
        },
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": "unused",
            "scheduler": {"name": "ddpm", "num_train_timesteps": 10},
        },
    }
    trainer = ControlNetTrainer(
        config=cfg,
        model_override=_FakeControlNet(),
        base_unet_override=_FakeBaseUNet(),
        noise_override=_FakeDDPMNoise(),
    )
    trainer._setup([{"target": torch.ones(1, 4, 4), "image": torch.ones(1, 4, 4)}], val_dataset=None, resume=None)
    state = trainer._build_state(epoch=1, metrics={"loss": 0.1})
    payload = trainer._build_checkpoint_dict(state)
    assert "base_only" not in payload["model"]


def test_load_frozen_base_unet_uses_payload_model_key(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {"trainer": "diffusion"},
        "model": {"model_type": "diffusion"},
    }
    load_calls: list[str] = []

    monkeypatch.setattr("models.controlnet.init_utils.load_run_config", lambda _ckpt_dir: cfg)
    monkeypatch.setattr("models.controlnet.init_utils.resolve_checkpoint", lambda _ckpt_dir, _model_type: tmp_path / "diff.pt")

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.0))

        def load_state_dict(self, state_dict, strict: bool = True):
            load_calls.append("loaded")
            assert "model_state" not in state_dict
            assert state_dict == {"param": torch.tensor(1.0)}
            return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr("models.controlnet.init_utils.ModelFactory.build", lambda _cfg: _FakeModel())
    monkeypatch.setattr(
        "models.controlnet.init_utils.torch.load",
        lambda *_args, **_kwargs: {"model": {"param": torch.tensor(1.0)}},
    )

    model = __import__("models.controlnet.init_utils", fromlist=["load_frozen_base_unet"]).load_frozen_base_unet(tmp_path, torch.device("cpu"))
    assert isinstance(model, torch.nn.Module)
    assert load_calls == ["loaded"]


def test_controlnet_run_step_does_not_mutate_model_train_eval_state(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "output_dir": str(tmp_path / "controlnet_mode"),
            "validate": False,
            "use_amp": False,
            "scheduler": "ddpm",
            "num_train_timesteps": 10,
        },
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": "unused",
            "scheduler": {"name": "ddpm", "num_train_timesteps": 10},
        },
    }
    trainer = ControlNetTrainer(
        config=cfg,
        model_override=_FakeControlNet(),
        base_unet_override=_FakeBaseUNet(),
        noise_override=_FakeDDPMNoise(),
    )
    batch = {"target": torch.ones(1, 1, 4, 4), "image": torch.ones(1, 1, 4, 4)}
    trainer._setup([batch], val_dataset=None, resume=None)

    assert trainer.model is not None
    trainer.model.eval()
    trainer._run_step(batch, train=True)
    assert trainer.model.training is False

    trainer.model.train()
    trainer._run_step(batch, train=False)
    assert trainer.model.training is True
