from __future__ import annotations

import warnings

import pipelines.train.diffusion_lib as diffusion_lib
import pipelines.train.flow_matching_lib as flow_matching_lib


def _assert_wrapper_dispatch(module, model_type: str, monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _DummyTrainer:
        @classmethod
        def from_config(cls, cfg):
            calls["cfg"] = cfg
            return cls()

        def fit(self, dataset, val_dataset=None, resume=None):
            calls["fit"] = (dataset, val_dataset, resume)

    monkeypatch.setattr(module, "load_json_config", lambda path: {"model": {"model_type": model_type}})
    monkeypatch.setattr(module.TRAINER_REGISTRY, "get", lambda key: _DummyTrainer if key == model_type else None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.train(dataset=[1], json_path="cfg.json", val_dataset=[2], resume="resume.pt")

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert calls["fit"] == ([1], [2], "resume.pt")


def test_diffusion_wrapper_dispatches(monkeypatch) -> None:
    _assert_wrapper_dispatch(diffusion_lib, "diffusion", monkeypatch)


def test_flow_matching_wrapper_dispatches(monkeypatch) -> None:
    _assert_wrapper_dispatch(flow_matching_lib, "flow_matching", monkeypatch)
