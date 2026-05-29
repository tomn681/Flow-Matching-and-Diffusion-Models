from __future__ import annotations

import pytest

from models.factory import MODEL_BUILD_STRATEGY, ModelFactory
from models.generators.diffusionfactory import DiffusionUNetFactory
from models.generators.vaefactory import VAEFactory
from models.registry import MODEL_REGISTRY


def test_model_registry_has_expected_entries() -> None:
    keys = set(MODEL_REGISTRY.list())
    assert {"kl_vae", "vq_vae", "efficient_unet", "diffusers_unet", "condition_unet"}.issubset(keys)


def test_model_factory_routes_to_vae(monkeypatch) -> None:
    sentinel = object()

    def _fake_build_vae(model_cfg: dict):
        assert model_cfg["model_type"] == "vae"
        return sentinel

    monkeypatch.setattr(ModelFactory, "_build_vae", staticmethod(_fake_build_vae))
    out = ModelFactory.build({"model": {"model_type": "vae"}})
    assert out is sentinel


def test_model_factory_routes_to_unet(monkeypatch) -> None:
    sentinel = object()

    def _fake_build_unet(model_cfg: dict, *, conditioning: str | None = None, channels: int | None = None):
        assert model_cfg["model_type"] == "diffusion"
        assert conditioning == "attention"
        assert channels == 2
        return sentinel

    monkeypatch.setattr(ModelFactory, "_build_unet", staticmethod(_fake_build_unet))
    out = ModelFactory.build({"model": {"model_type": "diffusion"}}, conditioning="attention", channels=2)
    assert out is sentinel


def test_model_factory_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model_type"):
        ModelFactory.build({"model": {"model_type": "unknown"}})


def test_model_build_strategy_contains_phase_i_types() -> None:
    keys = set(MODEL_BUILD_STRATEGY.keys())
    assert {
        "vae",
        "diffusion",
        "flow_matching",
        "latent_diffusion",
        "latent_flow_matching",
        "latent_rectified_flow",
        "consistency",
        "edm",
        "rectified_flow",
        "reflow",
    }.issubset(keys)


def test_legacy_vae_factory_delegates_to_unified_model_factory(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"model": {"model_type": "vae"}}')
    sentinel = object()

    def _fake_build(cfg: dict, *, conditioning=None, channels=None):
        assert cfg["model"]["model_type"] == "vae"
        assert conditioning is None
        assert channels is None
        return sentinel

    monkeypatch.setattr("models.generators.vaefactory.ModelFactory.build", _fake_build)
    out = VAEFactory().build_from_json(cfg_path)
    assert out is sentinel


def test_legacy_diffusion_factory_delegates_to_unified_model_factory(monkeypatch) -> None:
    sentinel = object()

    def _fake_build(cfg: dict, *, conditioning=None, channels=None):
        assert cfg["model"]["model_type"] == "diffusion"
        assert cfg["model"]["unet"]["in_channels"] == 1
        assert conditioning == "concatenate"
        assert channels == 1
        return sentinel

    monkeypatch.setattr("models.generators.diffusionfactory.ModelFactory.build", _fake_build)
    out = DiffusionUNetFactory().build({"in_channels": 1}, conditioning="concatenate", channels=1)
    assert out is sentinel
