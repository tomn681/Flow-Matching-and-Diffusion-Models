from __future__ import annotations

from pathlib import Path

import torch

from sampling import ControlNetSampler, SAMPLER_REGISTRY
from sampling.controlnet_sampler import _build_controlnet_inference_pipeline, _run_controlnet_inference


def test_controlnet_sampler_in_sampler_registry() -> None:
    assert SAMPLER_REGISTRY.get("controlnet") is ControlNetSampler


def test_controlnet_sampler_loads_both_models(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {},
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": str(tmp_path / "base_run"),
            "scheduler": {},
        },
    }
    calls: list[str] = []

    class _Tiny(torch.nn.Module):
        def forward(self, *args, **kwargs):
            _ = args, kwargs
            return torch.zeros(1, 1, 4, 4)

    monkeypatch.setattr("sampling.controlnet_sampler.load_frozen_base_unet", lambda *args, **kwargs: calls.append("base") or _Tiny())
    monkeypatch.setattr("sampling.controlnet_sampler._load_controlnet_model", lambda *args, **kwargs: calls.append("controlnet") or _Tiny())
    monkeypatch.setattr("sampling.controlnet_sampler.build_scheduler", lambda *args, **kwargs: ("scheduler", 25))

    pipe, steps = _build_controlnet_inference_pipeline(cfg=cfg, ckpt_dir=tmp_path, device=torch.device("cpu"))
    assert steps == 25
    assert calls == ["base", "controlnet"]
    assert pipe.unet is not None
    assert pipe.controlnet is not None


def test_run_controlnet_inference_smoke_saves_predicted_outputs(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {},
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": str(tmp_path / "base_run"),
            "scheduler": {},
        },
        "sampling": {},
    }

    class _Dataset:
        target_key = "target"
        conditioning_key = "image"
        data = [{"img_id": "a", "img_path": "a.dcm"}]

    sample = {
        "target": torch.zeros(1, 8, 8),
        "image": torch.ones(1, 8, 8),
        "attn_cond": torch.full((1, 8, 8), 2.0),
        "img_id": "a",
        "img_path": "a.dcm",
    }
    dataset = _Dataset()
    saved: list[tuple[str, str]] = []

    class _Pipe:
        def generate(self, inputs):
            assert inputs.sample_shape == (1, 1, 8, 8)
            assert inputs.controlnet_cond is not None
            assert tuple(inputs.controlnet_cond.shape) == (1, 1, 8, 8)
            assert inputs.conditioning_batch is not None
            return torch.full(inputs.sample_shape, 0.5)

    monkeypatch.setattr("sampling.controlnet_sampler.load_run_config", lambda _ckpt: cfg)
    monkeypatch.setattr("sampling.controlnet_sampler.build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("sampling.controlnet_sampler.resolve_sample_indices", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        "sampling.controlnet_sampler.progress_batches",
        lambda _dataset, _batch_size, _desc, indices=None: [([0], [sample])],
    )
    monkeypatch.setattr("sampling.controlnet_sampler._build_controlnet_inference_pipeline", lambda **kwargs: (_Pipe(), 7))
    monkeypatch.setattr(
        "sampling.controlnet_sampler.save_output_tensor",
        lambda dataset, row, key, tensor, output_root: saved.append((str(key), str(output_root))),
    )

    _run_controlnet_inference(
        ckpt_dir=tmp_path,
        save=True,
        output_dir=str(tmp_path / "outputs"),
        batch_size=1,
        evaluate=False,
    )

    assert saved
    assert saved[0][0] == "target"


def test_run_controlnet_evaluate_smoke_writes_metrics(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {},
        "model": {
            "model_type": "controlnet",
            "base_unet_checkpoint": str(tmp_path / "base_run"),
            "scheduler": {},
        },
        "sampling": {},
    }

    class _Dataset:
        target_key = "target"
        conditioning_key = "image"
        data = [{"img_id": "a", "img_path": "a.dcm"}]

    sample = {
        "target": torch.zeros(1, 8, 8),
        "image": torch.ones(1, 8, 8),
        "img_id": "a",
        "img_path": "a.dcm",
    }
    dataset = _Dataset()

    class _Pipe:
        def generate(self, inputs):
            return torch.zeros(inputs.sample_shape)

    monkeypatch.setattr("sampling.controlnet_sampler.load_run_config", lambda _ckpt: cfg)
    monkeypatch.setattr("sampling.controlnet_sampler.build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("sampling.controlnet_sampler.resolve_sample_indices", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        "sampling.controlnet_sampler.progress_batches",
        lambda _dataset, _batch_size, _desc, indices=None: [([0], [sample])],
    )
    monkeypatch.setattr("sampling.controlnet_sampler._build_controlnet_inference_pipeline", lambda **kwargs: (_Pipe(), 7))
    monkeypatch.setattr(
        "sampling.controlnet_sampler.create_experiment_dir",
        lambda **kwargs: (tmp_path / "eval_run").mkdir(parents=True, exist_ok=True) or (tmp_path / "eval_run"),
    )
    monkeypatch.setattr("sampling.controlnet_sampler.write_eval_metrics", lambda root, row: root / "metrics.csv")
    monkeypatch.setattr("sampling.controlnet_sampler.append_per_image_eval_metrics", lambda root, rows: root / "per_image.csv")

    _run_controlnet_inference(
        ckpt_dir=tmp_path,
        batch_size=1,
        evaluate=True,
    )
