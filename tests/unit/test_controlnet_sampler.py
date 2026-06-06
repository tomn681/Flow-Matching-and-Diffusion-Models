from __future__ import annotations

from pathlib import Path

import torch

from sampling import ControlNetSampler, SAMPLER_REGISTRY
from sampling.controlnet_sampler import _build_controlnet_inference_pipeline


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

