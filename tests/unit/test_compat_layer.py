from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import compat.legacy_training as legacy_training
from compat.legacy_config import adapt_legacy_config_v1
import compat.legacy_samplers as legacy_samplers
from compat.legacy_samplers import DiffusionHandler, FlowMatchingHandler, VAEHandler


def test_adapt_legacy_config_v1_normalizes_and_ensures_sections() -> None:
    cfg = {"training": {"num_epochs": 2}, "model": None}
    adapted = adapt_legacy_config_v1(cfg)
    assert adapted["training"]["epochs"] == 2
    assert isinstance(adapted["model"], dict)
    assert isinstance(adapted["training"], dict)


def test_legacy_training_wrapper_delegates_to_registry(monkeypatch) -> None:
    calls = {}

    class _DummyTrainer:
        @classmethod
        def from_config(cls, cfg):
            calls["cfg"] = cfg
            return cls()

        def fit(self, dataset, val_dataset=None, resume=None):
            calls["fit"] = (dataset, val_dataset, resume)

    monkeypatch.setattr(legacy_training, "load_json_config", lambda path: {"model": {"model_type": "vae"}})
    monkeypatch.setattr(legacy_training.TRAINER_REGISTRY, "get", lambda key: _DummyTrainer if key == "vae" else None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        legacy_training.train_vae(dataset=[1], json_path="cfg.json", val_dataset=[2], resume="r.pt")
    assert any(issubclass(item.category, DeprecationWarning) for item in w)
    assert any("v2.0" in str(item.message) for item in w)
    assert calls["fit"] == ([1], [2], "r.pt")


def test_legacy_sampler_aliases_emit_deprecation_warning(monkeypatch, tmp_path: Path) -> None:
    class _DummySampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def encode(self):
            return None

        def decode(self):
            return None

        def sample(self):
            return None

        def evaluate(self):
            return None

        def build_tensor_cache(self):
            return None

        def debug_compare(self):
            return None

    monkeypatch.setattr(legacy_samplers.SAMPLER_REGISTRY, "get", lambda key: _DummySampler)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = VAEHandler(ckpt_dir=tmp_path)
        _ = DiffusionHandler(ckpt_dir=tmp_path)
        _ = FlowMatchingHandler(ckpt_dir=tmp_path)
    dep_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
    assert len(dep_warnings) >= 3
    assert all("v2.0" in str(item.message) for item in dep_warnings)


def test_compat_package_reexports_warn_on_attribute_access() -> None:
    compat_pkg = importlib.import_module("compat")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = compat_pkg.train_vae
        _ = compat_pkg.VAEHandler
    dep_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
    assert len(dep_warnings) >= 2
    assert any("compat.train_vae" in str(item.message) for item in dep_warnings)
    assert any("compat.VAEHandler" in str(item.message) for item in dep_warnings)


def test_pipelines_compat_aggregators_warn_on_attribute_access() -> None:
    train_pkg = importlib.import_module("pipelines.train")
    handlers_pkg = importlib.import_module("pipelines.samplers.handlers")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = train_pkg.train_vae
        _ = handlers_pkg.VAEHandler
    dep_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
    assert len(dep_warnings) >= 2
    assert any("pipelines.train.train_vae" in str(item.message) for item in dep_warnings)
    assert any("pipelines.samplers.handlers.VAEHandler" in str(item.message) for item in dep_warnings)
