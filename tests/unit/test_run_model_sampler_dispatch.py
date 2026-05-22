from __future__ import annotations

from pathlib import Path

import run_model


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
            },
        )(),
    )

    run_model.main()
    assert called["mode"] == "sample"
