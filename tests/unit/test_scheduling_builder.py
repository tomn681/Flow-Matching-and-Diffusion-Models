from __future__ import annotations

import pytest

from scheduling.builder import build_scheduler, resolve_scheduler_override


def test_build_scheduler_ddpm_default() -> None:
    scheduler, steps = build_scheduler({}, {})
    assert hasattr(scheduler, "config")
    assert hasattr(scheduler.config, "num_train_timesteps")


def test_build_scheduler_with_name() -> None:
    scheduler, steps = build_scheduler({"name": "ddpm"}, {})
    assert scheduler.config.num_train_timesteps == 1000


def test_build_scheduler_custom_timesteps() -> None:
    scheduler, steps = build_scheduler({"name": "ddpm", "num_train_timesteps": 500}, {})
    assert scheduler.config.num_train_timesteps == 500


def test_build_scheduler_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler({"name": "nonexistent_scheduler"}, {})


def test_resolve_scheduler_override_ddpm() -> None:
    result = resolve_scheduler_override("ddpm")
    assert result == {"name": "ddpm"}


def test_resolve_scheduler_override_none() -> None:
    assert resolve_scheduler_override(None) is None
    assert resolve_scheduler_override("") is None


def test_resolve_scheduler_override_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler override"):
        resolve_scheduler_override("totally_unknown_name")

