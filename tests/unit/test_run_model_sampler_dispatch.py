from __future__ import annotations

from pathlib import Path

import run_model
from sampling.base import BaseSampler


def test_run_model_dispatches_to_sampler_registry(monkeypatch, tmp_path: Path) -> None:
    called = {"request": None}

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "vae"}})
    monkeypatch.setattr(
        "sampling.engine.SamplingEngine.run",
        lambda self, request: called.__setitem__("request", request),
    )
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
                "save_diff_map": True,
                "diff_amplify": 6.0,
                "save_tensor_cache": False,
                "num_pairs": None,
                "use_ema": True,
            },
        )(),
    )

    run_model.main()
    assert called["request"].mode == "sample"
    assert called["request"].save_diff_map is True
    assert called["request"].diff_amplify == 6.0
    assert called["request"].use_ema is True


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
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": None,
                "use_ema": False,
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
    monkeypatch.setattr("sampling.engine.SamplingEngine.resolve_sampler_cls", lambda self, model_type: _DecodeOnlySampler)
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
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": None,
                "use_ema": False,
            },
        )(),
    )
    try:
        run_model.main()
        raise AssertionError("Expected ValueError for unsupported sampler capability.")
    except ValueError as exc:
        assert "not implemented by sampler" in str(exc)


def test_run_model_dispatches_generate_reflow_pairs_mode(monkeypatch, tmp_path: Path) -> None:
    called = {"request": None}

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "flow_matching"}})
    monkeypatch.setattr(
        "sampling.engine.SamplingEngine.run",
        lambda self, request: called.__setitem__("request", request),
    )
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
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": 10,
                "use_ema": False,
            },
        )(),
    )

    run_model.main()
    assert called["request"].mode == "generate_reflow_pairs"
    assert called["request"].num_pairs == 10


def test_run_model_dispatches_controlnet_sampler(monkeypatch, tmp_path: Path) -> None:
    called = {"request": None}

    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "controlnet"}})
    monkeypatch.setattr(
        "sampling.engine.SamplingEngine.run",
        lambda self, request: called.__setitem__("request", request),
    )
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
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": None,
                "use_ema": False,
            },
        )(),
    )

    run_model.main()
    assert called["request"].mode == "sample"


def test_run_model_dispatches_multi_checkpoint_batch(monkeypatch, tmp_path: Path) -> None:
    called = {"requests": None}
    ckpt_a = tmp_path / "a"
    ckpt_b = tmp_path / "b"

    monkeypatch.setattr(run_model, "load_run_config", lambda path: {"model": {"model_type": "vae" if Path(path).name == "a" else "diffusion"}})
    monkeypatch.setattr(
        "sampling.engine.SamplingEngine.run_many",
        lambda self, requests: called.__setitem__("requests", requests),
    )
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "ckpt_dir": ckpt_a,
                "ckpt_dirs": [ckpt_a, ckpt_b],
                "mode": "evaluate",
                "data_txt": None,
                "save": True,
                "output_dir": str(tmp_path / "outs"),
                "batch_size": 4,
                "device": None,
                "seed": 42,
                "timestep": None,
                "num_samples": None,
                "num_inference_steps": None,
                "start_step": None,
                "last_n_steps": None,
                "cfg_rescale": 0.25,
                "scheduler": None,
                "save_input": False,
                "save_conditioning": False,
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": None,
                "use_ema": False,
            },
        )(),
    )

    run_model.main()
    assert called["requests"] is not None
    assert len(called["requests"]) == 2
    assert all(request.mode == "evaluate" for request in called["requests"])
    assert called["requests"][0].cfg_rescale == 0.25
    assert called["requests"][0].output_dir.endswith("/a")
    assert called["requests"][1].output_dir.endswith("/b")


def test_run_model_generate_reflow_pairs_smoke_writes_z0_z1(monkeypatch, tmp_path: Path) -> None:
    import torch

    class _FakeSampler:
        def __init__(self, **kwargs) -> None:
            self.output_dir = kwargs.get("output_dir")
            self.num_pairs = int(kwargs.get("num_pairs") or 2)

        def generate_reflow_pairs(self) -> None:
            out = Path(self.output_dir or (tmp_path / "reflow_pairs"))
            out.mkdir(parents=True, exist_ok=True)
            for i in range(self.num_pairs):
                torch.save(
                    {"z0": torch.randn(1, 4, 4), "z1": torch.randn(1, 4, 4)},
                    out / f"{i:08d}.pt",
                )

    out_dir = tmp_path / "pairs_out"
    monkeypatch.setattr(run_model, "load_run_config", lambda _: {"model": {"model_type": "flow_matching"}})
    monkeypatch.setattr("sampling.engine.SamplingEngine.resolve_sampler_cls", lambda self, model_type: _FakeSampler)
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
                "output_dir": str(out_dir),
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
                "save_diff_map": False,
                "diff_amplify": 5.0,
                "save_tensor_cache": False,
                "num_pairs": 3,
                "use_ema": False,
            },
        )(),
    )

    run_model.main()
    files = sorted(out_dir.glob("*.pt"))
    assert len(files) == 3
    payload = torch.load(files[0], map_location="cpu")
    assert set(payload.keys()) == {"z0", "z1"}


def test_run_model_interrupt_label_is_mode_specific() -> None:
    assert run_model._interrupt_label("sample") == "Sampling"
    assert run_model._interrupt_label("evaluate") == "Evaluation"
    assert run_model._interrupt_label("build_tensor_cache") == "Tensor cache build"
