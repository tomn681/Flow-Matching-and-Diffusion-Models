from __future__ import annotations

from pathlib import Path

import run_model
from sampling.base import BaseSampler


def test_run_model_dispatches_to_sampler_registry(monkeypatch, tmp_path: Path) -> None:
    called = {"mode": None}

    class _DummySampler:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def encode(self) -> None:
            called["mode"] = "encode"

        def decode(self) -> None:
            called["mode"] = "decode"

        def sample(self) -> None:
            called["mode"] = "sample"

        def evaluate(self) -> None:
            called["mode"] = "evaluate"

        def build_tensor_cache(self) -> None:
            called["mode"] = "build_tensor_cache"

        def debug_compare(self) -> None:
            called["mode"] = "debug_compare"

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "vae"}})
    monkeypatch.setattr(run_model.SAMPLER_REGISTRY, "get", lambda key: _DummySampler if key == "vae" else None)
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "ckpt_dir": tmp_path,
                "mode": "sample",
                "data_txt": None,
                "save": False,
                "output_dir": None,
                "batch_size": 4,
                "device": None,
                "seed": 42,
                "timestep": None,
                "num_samples": None,
                "num_inference_steps": None,
                "start_step": None,
                "last_n_steps": None,
                "scheduler": None,
                "save_input": False,
                "save_conditioning": False,
                "save_tensor_cache": False,
                "num_pairs": None,
            },
        )(),
    )

    run_model.main()
    assert called["mode"] == "sample"


def test_run_model_rejects_unsupported_mode_for_latent_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "latent_diffusion"}})
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "ckpt_dir": tmp_path,
                "mode": "evaluate",
                "data_txt": None,
                "save": False,
                "output_dir": None,
                "batch_size": 4,
                "device": None,
                "seed": 42,
                "timestep": None,
                "num_samples": None,
                "num_inference_steps": None,
                "start_step": None,
                "last_n_steps": None,
                "scheduler": None,
                "save_input": False,
                "save_conditioning": False,
                "save_tensor_cache": False,
                "num_pairs": None,
            },
        )(),
    )
    try:
        run_model.main()
        raise AssertionError("Expected ValueError for unsupported latent mode.")
    except ValueError as exc:
        assert "Supported modes" in str(exc)


def test_run_model_rejects_sampler_without_mode_capability(monkeypatch, tmp_path: Path) -> None:
    class _DecodeOnlySampler(BaseSampler):
        def decode(self) -> None:
            return None

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "vae"}})
    monkeypatch.setattr(
        run_model.SAMPLER_REGISTRY,
        "get",
        lambda key: _DecodeOnlySampler if key == "vae" else None,
    )
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "ckpt_dir": tmp_path,
                "mode": "evaluate",
                "data_txt": None,
                "save": False,
                "output_dir": None,
                "batch_size": 4,
                "device": None,
                "seed": 42,
                "timestep": None,
                "num_samples": None,
                "num_inference_steps": None,
                "start_step": None,
                "last_n_steps": None,
                "scheduler": None,
                "save_input": False,
                "save_conditioning": False,
                "save_tensor_cache": False,
                "num_pairs": None,
            },
        )(),
    )
    try:
        run_model.main()
        raise AssertionError("Expected ValueError for unsupported sampler capability.")
    except ValueError as exc:
        assert "not implemented by sampler" in str(exc)


def test_run_model_dispatches_generate_reflow_pairs_mode(monkeypatch, tmp_path: Path) -> None:
    called = {"mode": None, "num_pairs": None}

    class _DummySampler:
        def __init__(self, **kwargs) -> None:
            called["num_pairs"] = kwargs.get("num_pairs")

        def generate_reflow_pairs(self) -> None:
            called["mode"] = "generate_reflow_pairs"

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "flow_matching"}})
    monkeypatch.setattr(run_model.SAMPLER_REGISTRY, "get", lambda key: _DummySampler if key == "flow_matching" else None)
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "ckpt_dir": tmp_path,
                "mode": "generate_reflow_pairs",
                "data_txt": None,
                "save": False,
                "output_dir": None,
                "batch_size": 4,
                "device": None,
                "seed": 42,
                "timestep": None,
                "num_samples": None,
                "num_inference_steps": None,
                "start_step": None,
                "last_n_steps": None,
                "scheduler": None,
                "save_input": False,
                "save_conditioning": False,
                "save_tensor_cache": False,
                "num_pairs": 10,
            },
        )(),
    )

    run_model.main()
    assert called["mode"] == "generate_reflow_pairs"
    assert called["num_pairs"] == 10
