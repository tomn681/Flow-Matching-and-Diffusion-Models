from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import EMAModel
from training.ema import apply_ema_state_to_model
from training.base import BaseTrainer
from core.types import TrainingState


class _EMAIntegrationTrainer(BaseTrainer):
    def _build_model(self):
        return nn.Linear(1, 1)

    def _build_optimizer(self):
        if self.model is None:
            raise RuntimeError("model not initialized")
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def _training_step(self, batch, *, epoch):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("trainer not initialized")
        x = batch["target"].view(-1, 1).to(self.device)
        pred = self.model(x)
        loss = (pred ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        self._backward(loss)
        self._step_optimizers(self.optimizer)
        return {"loss": float(loss.detach().item())}

    def _validation_step(self, batch, *, epoch):
        if self.model is None:
            raise RuntimeError("trainer not initialized")
        x = batch["target"].view(-1, 1).to(self.device)
        pred = self.model(x)
        loss = (pred ** 2).mean()
        return {"loss": float(loss.detach().item())}


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        return {"target": torch.tensor([1.0], dtype=torch.float32)}


def test_ema_weights_converge_toward_model_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMAModel(model, decay=0.5, use_warmup=False)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model)

    ema_value = float(ema.shadow_params["weight"].item())
    assert ema_value == torch.tensor(0.75).item()


def test_ema_state_dict_roundtrip_preserves_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = EMAModel(model, decay=0.9)
    state = ema.state_dict()

    other = nn.Linear(1, 1, bias=False)
    loaded = EMAModel(other, decay=0.1)
    loaded.load_state_dict(state)

    assert loaded.decay == 0.9
    assert torch.allclose(loaded.shadow_params["weight"], ema.shadow_params["weight"])


def test_ema_warmup_decay_starts_below_target_decay() -> None:
    model = nn.Linear(1, 1, bias=False)
    ema = EMAModel(model, decay=0.99, use_warmup=True)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model)
    assert ema.num_updates == 1
    assert float(ema.shadow_params["weight"].item()) > 0.0
    assert ema._effective_decay() < 0.99


def test_ema_copy_to_matches_shadow_parameters() -> None:
    src = nn.Linear(1, 1, bias=False)
    dst = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        src.weight.fill_(3.0)
        dst.weight.fill_(0.0)
    ema = EMAModel(src, decay=0.99)
    ema.copy_to(dst)
    assert torch.allclose(dst.weight, src.weight)


def test_ema_average_parameters_restores_original_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMAModel(model, decay=0.9, use_warmup=False)
    ema.shadow_params["weight"].fill_(2.0)

    with ema.average_parameters(model):
        assert torch.allclose(model.weight, torch.tensor([[2.0]]))

    assert torch.allclose(model.weight, torch.tensor([[0.0]]))


def test_apply_ema_state_to_model_overwrites_named_parameters() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    applied = apply_ema_state_to_model(
        model,
        {"shadow_params": {"weight": torch.tensor([[3.0]])}},
    )
    assert applied is True
    assert torch.allclose(model.weight, torch.tensor([[3.0]]))


def test_base_trainer_checkpoint_contains_ema_state(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_ema"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "ema_decay": 0.9,
        },
        "model": {},
    }
    trainer = _EMAIntegrationTrainer(cfg, callbacks=[])
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    assert trainer.ema_model is not None
    ts = TrainingState(
        epoch=1,
        global_step=trainer.global_step,
        model_state=trainer.model.state_dict(),
        optimizer_state=trainer.optimizer.state_dict(),
        metrics={"loss": 0.0},
        extra={
            "scheduler": None,
            "scaler": None,
            "ema": trainer.ema_model.state_dict(),
        },
    )
    payload = trainer._build_checkpoint_dict(ts)
    assert payload.get("ema") is not None
