from __future__ import annotations

import torch
import torch.nn as nn

from core.types import NoisyBatch
from training import DiffusionTrainer


class _CaptureUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.last_input = None
        self.last_context = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        self.last_input = x.detach().clone()
        self.last_context = None if context_ca is None else context_ca.detach().clone()
        pred = x[:, :1, ...] if x.size(1) > 1 else x
        return pred * self.weight


class _FakeNoise:
    def __call__(self, clean: torch.Tensor, device: torch.device):
        noisy = clean.to(device)
        target = torch.zeros_like(noisy)
        timesteps = torch.zeros(clean.size(0), device=device, dtype=torch.long)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)


def _make_trainer(*, conditioning: str, conditioning_dropout: float) -> DiffusionTrainer:
    cfg = {
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "use_amp": False,
            "manual_device": "cpu",
            "conditioning": conditioning,
            "conditioning_dropout": conditioning_dropout,
            "epochs": 1,
            "save_every": 1,
        },
        "model": {"model_type": "diffusion", "conditioning": conditioning, "scheduler": {}},
    }
    trainer = DiffusionTrainer(cfg, callbacks=[])
    trainer.device = torch.device("cpu")
    trainer.model = _CaptureUNet()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.noise_process = _FakeNoise()
    return trainer


def test_cfg_dropout_attention_all_dropped() -> None:
    trainer = _make_trainer(conditioning="attention", conditioning_dropout=1.0)
    batch = {"target": torch.ones(8, 1, 4, 4), "image": torch.ones(8, 1, 4, 4)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.model.last_context is not None
    assert torch.allclose(trainer.model.last_context, torch.zeros_like(trainer.model.last_context))


def test_cfg_dropout_attention_none_dropped() -> None:
    trainer = _make_trainer(conditioning="attention", conditioning_dropout=0.0)
    batch = {"target": torch.ones(8, 1, 4, 4), "image": torch.ones(8, 1, 4, 4)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.model.last_context is not None
    assert torch.count_nonzero(trainer.model.last_context).item() > 0


def test_cfg_dropout_concatenate_zeroes_concat_channels() -> None:
    trainer = _make_trainer(conditioning="concatenate", conditioning_dropout=1.0)
    batch = {"target": torch.ones(8, 1, 4, 4), "image": torch.ones(8, 1, 4, 4)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.model.last_input is not None
    cond_part = trainer.model.last_input[:, 1:, ...]
    assert torch.allclose(cond_part, torch.zeros_like(cond_part))


def test_cfg_dropout_attention_statistical_rate() -> None:
    trainer = _make_trainer(conditioning="attention", conditioning_dropout=0.1)
    batch = {"target": torch.ones(1000, 1, 2, 2), "image": torch.ones(1000, 1, 2, 2)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.model.last_context is not None
    per_sample_sum = trainer.model.last_context.abs().flatten(1).sum(dim=1)
    dropped = int((per_sample_sum == 0).sum().item())
    assert 70 <= dropped <= 130
