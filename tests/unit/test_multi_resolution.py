from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from configs.training import MultiResolutionStageConfig
from training.callbacks import MultiResolutionCallback
from training.multi_resolution import (
    StepwiseResolutionSchedule,
    _check_multi_resolution_compatibility,
    build_resolution_schedule,
)


def test_stepwise_schedule_returns_correct_resolution_per_epoch() -> None:
    schedule = StepwiseResolutionSchedule(
        stages=[
            MultiResolutionStageConfig(start_epoch=0, resolution=64),
            MultiResolutionStageConfig(start_epoch=5, resolution=128),
        ]
    )
    assert schedule.current_resolution(0) == 64
    assert schedule.current_resolution(4) == 64
    assert schedule.current_resolution(5) == 128


def test_stepwise_schedule_before_first_stage_raises() -> None:
    schedule = StepwiseResolutionSchedule(
        stages=[MultiResolutionStageConfig(start_epoch=2, resolution=64)]
    )
    with pytest.raises(ValueError, match="before first stage"):
        schedule.current_resolution(1)


def test_build_resolution_schedule_returns_none_when_absent() -> None:
    assert build_resolution_schedule({"training": {}}) is None


def test_build_resolution_schedule_raises_on_unsorted_stages() -> None:
    cfg = {
        "training": {
            "multi_resolution": [
                {"start_epoch": 5, "resolution": 128},
                {"start_epoch": 0, "resolution": 64},
            ]
        }
    }
    with pytest.raises(ValueError, match="sorted"):
        build_resolution_schedule(cfg)


def test_build_resolution_schedule_raises_on_missing_epoch_zero() -> None:
    cfg = {
        "training": {
            "multi_resolution": [
                {"start_epoch": 1, "resolution": 64},
            ]
        }
    }
    with pytest.raises(ValueError, match="epoch 0"):
        build_resolution_schedule(cfg)


def test_multi_resolution_callback_rebuilds_dataloader_on_change() -> None:
    class _Trainer:
        def __init__(self):
            self._resolution_schedule = StepwiseResolutionSchedule(
                stages=[
                    MultiResolutionStageConfig(start_epoch=0, resolution=64),
                    MultiResolutionStageConfig(start_epoch=2, resolution=128),
                ]
            )
            self.calls: list[int] = []

        def _rebuild_dataloaders(self, *, target_resolution: int, **kwargs):
            _ = kwargs
            self.calls.append(target_resolution)

    trainer = _Trainer()
    cb = MultiResolutionCallback()
    cb.on_epoch_start(epoch=0, trainer=trainer)
    cb.on_epoch_start(epoch=1, trainer=trainer)
    cb.on_epoch_start(epoch=2, trainer=trainer)
    cb.on_epoch_start(epoch=3, trainer=trainer)
    assert trainer.calls == [64, 128]


def test_compatibility_check_warns_on_attention_resolution_mismatch(caplog) -> None:
    from models.unet.base import BaseUNetND

    class _DummyUNet(BaseUNetND):
        def __init__(self):
            super().__init__()
            self.attention_resolutions = (3,)

        def _build_time_embedding(self, t, x):
            _ = x
            return t.float().unsqueeze(1)

        def _run_network(self, x, emb, context_ca=None, *, attention_mask=None, encoder_attention_mask=None):
            _ = emb, context_ca, attention_mask, encoder_attention_mask
            return x

    schedule = StepwiseResolutionSchedule(
        stages=[MultiResolutionStageConfig(start_epoch=0, resolution=64)]
    )
    caplog.set_level(logging.WARNING)
    _check_multi_resolution_compatibility(_DummyUNet(), schedule)
    assert any("may miss configured attention_resolutions" in rec.message for rec in caplog.records)


def test_compatibility_check_raises_for_non_fcn_model() -> None:
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Parameter(torch.zeros(1, 4))

    schedule = StepwiseResolutionSchedule(
        stages=[MultiResolutionStageConfig(start_epoch=0, resolution=64)]
    )
    with pytest.raises(NotImplementedError, match="fully convolutional"):
        _check_multi_resolution_compatibility(_Bad(), schedule)
