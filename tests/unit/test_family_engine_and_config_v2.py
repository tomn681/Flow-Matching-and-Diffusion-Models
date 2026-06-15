from __future__ import annotations

import json
from pathlib import Path

import pytest

from configs.v2 import resolve_config
from core.families import get_model_family, model_family_for_model_type
from core.plugin import RegistryHub, load_plugins
from sampling.engine import SamplingEngine, SamplingRequest


def test_builtin_model_families_resolve_by_model_type() -> None:
    diffusion = get_model_family("diffusion")
    assert diffusion.trainer_key == "diffusion"
    assert diffusion.sampler_key == "diffusion"
    assert diffusion.noise_family == "ddpm"

    latent = model_family_for_model_type("latent_flow_matching")
    assert latent is not None
    assert latent.key == "latent_flow_matching"
    assert latent.runtime_kind == "latent"
    assert latent.latent_capable is True


def test_load_plugins_calls_register_with_registry_hub(monkeypatch) -> None:
    called = {"hub": None}

    class _EP:
        name = "demo"

        def load(self):
            def _register(hub=None):
                called["hub"] = hub
                return "demo"

            return _register

    class _SelectableEPs:
        def select(self, *, group: str):
            assert group == "genlib.plugins"
            return [_EP()]

    monkeypatch.setattr("core.plugin.metadata.entry_points", lambda: _SelectableEPs())
    hub = RegistryHub(model_families=object())
    assert load_plugins(hub=hub) == ["demo"]
    assert called["hub"] is hub


def test_sampling_engine_resolves_family_sampler_and_runs(monkeypatch, tmp_path: Path) -> None:
    called = {"kwargs": None, "mode": None}

    class _DummySampler:
        def __init__(self, **kwargs) -> None:
            called["kwargs"] = kwargs

        def sample(self) -> None:
            called["mode"] = "sample"

    monkeypatch.setattr("sampling.engine.SAMPLER_REGISTRY.get", lambda key: _DummySampler if key == "diffusion" else None)
    engine = SamplingEngine()
    request = SamplingRequest(ckpt_dir=tmp_path, model_type="diffusion", mode="sample", use_ema=True)
    engine.run(request)

    assert called["mode"] == "sample"
    assert called["kwargs"]["use_ema"] is True


def test_resolve_config_supports_base_and_dotted_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text(
        json.dumps(
            {
                "model": {"model_type": "vae", "z_channels": 4},
                "training": {"epochs": 10, "batch_size": 2},
                "dataset": {"conditioning": False},
            }
        ),
        encoding="utf-8",
    )
    child.write_text(
        json.dumps(
            {
                "_base_": "base.json",
                "training": {"batch_size": 8},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_config(child, overrides=["training.epochs=20", "model.z_channels=8"])
    assert resolved["training"]["epochs"] == 20
    assert resolved["training"]["batch_size"] == 8
    assert resolved["model"]["z_channels"] == 8
    assert resolved["dataset"]["conditioning"] is False
    assert resolved["__config_path__"] == str(child)


def test_resolve_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": {"model_type": "vae"}, "training": {}, "surprise": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level config keys"):
        resolve_config(bad)
