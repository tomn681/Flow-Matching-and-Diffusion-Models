from __future__ import annotations

import pytest

from scheduling.builder import build_scheduler, resolve_scheduler_override
from scheduling.registry import SCHEDULER_REGISTRY


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


def test_build_scheduler_rejects_unsupported_params() -> None:
    with pytest.raises(ValueError, match="Unsupported scheduler params"):
        build_scheduler({"name": "ddpm", "params": {"not_a_real_param": 1}}, {})


def test_build_scheduler_accepts_top_level_allowlisted_params() -> None:
    scheduler, _ = build_scheduler({"name": "ddpm", "beta_schedule": "scaled_linear"}, {})
    assert scheduler.config.beta_schedule == "scaled_linear"


def test_build_flow_match_scheduler_accepts_shift() -> None:
    scheduler, _ = build_scheduler({"name": "flow_match_euler", "shift": 1.5}, {}, noise_family="flow_matching")
    assert float(scheduler.config.shift) == pytest.approx(1.5)


def test_build_scheduler_uses_solver_runtime_defaults() -> None:
    _, steps = build_scheduler({"name": "dpm_multistep"}, {})
    assert steps == 20


def test_build_scheduler_enables_karras_sigmas_for_diffusion_euler_family() -> None:
    scheduler, _ = build_scheduler({"name": "euler"}, {}, noise_family="diffusion")
    assert bool(scheduler.config.use_karras_sigmas) is True


def test_resolve_scheduler_override_ddpm() -> None:
    result = resolve_scheduler_override("ddpm")
    assert result == {"name": "ddpm"}


def test_resolve_scheduler_override_karras_alias() -> None:
    result = resolve_scheduler_override("dpmpp_karras")
    assert result["name"] == "dpm_multistep"
    assert result["params"]["use_karras_sigmas"] is True


def test_resolve_scheduler_override_none() -> None:
    assert resolve_scheduler_override(None) is None
    assert resolve_scheduler_override("") is None


def test_resolve_scheduler_override_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler override"):
        resolve_scheduler_override("totally_unknown_name")


def test_extended_scheduler_registry_keys_present() -> None:
    expected = {
        "pndm",
        "euler",
        "euler_ancestral",
        "heun",
        "lms",
        "kdpm2",
        "kdpm2_ancestral",
        "deis",
    }
    assert expected.issubset(set(SCHEDULER_REGISTRY.keys()))


def test_extended_schedulers_build_successfully() -> None:
    for key in [
        "pndm",
        "euler",
        "euler_ancestral",
        "heun",
        "lms",
        "kdpm2",
        "kdpm2_ancestral",
        "deis",
    ]:
        scheduler, steps = build_scheduler({"name": key}, {})
        assert hasattr(scheduler, "config")
        assert hasattr(scheduler.config, "num_train_timesteps")
        assert isinstance(steps, int)
