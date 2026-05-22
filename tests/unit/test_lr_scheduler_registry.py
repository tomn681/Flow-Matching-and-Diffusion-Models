from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scheduling.lr import LR_SCHEDULER_REGISTRY, build_lr_scheduler


def _optimizer() -> AdamW:
    param = torch.nn.Parameter(torch.tensor(1.0))
    return AdamW([param], lr=1e-3)


def test_lr_scheduler_registry_entries() -> None:
    assert LR_SCHEDULER_REGISTRY.list() == ["cosineannealinglr", "exponentiallr", "steplr"]


def test_build_lr_scheduler_from_string() -> None:
    scheduler = build_lr_scheduler(_optimizer(), {"scheduler": "cosineannealinglr"})
    assert isinstance(scheduler, CosineAnnealingLR)


def test_build_lr_scheduler_none_string_returns_none() -> None:
    scheduler = build_lr_scheduler(_optimizer(), {"scheduler": "none"})
    assert scheduler is None


def test_build_lr_scheduler_missing_returns_none() -> None:
    scheduler = build_lr_scheduler(_optimizer(), {})
    assert scheduler is None


def test_build_lr_scheduler_unknown_raises() -> None:
    try:
        build_lr_scheduler(_optimizer(), {"scheduler": "does_not_exist"})
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown scheduler name.")


def test_build_lr_scheduler_invalid_spec_type_raises() -> None:
    try:
        build_lr_scheduler(_optimizer(), {"scheduler": 123})
    except TypeError:
        return
    raise AssertionError("Expected TypeError for invalid scheduler spec type.")
