from __future__ import annotations

import torch

from pipelines.samplers.diffusion_like import _build_conditioning_batch, _resolve_conditioning_save_tensor, _run_decode


def test_build_conditioning_batch_chain_uses_explicit_keys() -> None:
    samples = [
        {
            "target": torch.zeros(1, 4, 4),
            "concat_cond": torch.ones(1, 4, 4),
            "attn_cond": torch.full((2, 4, 4), 3.0),
        },
        {
            "target": torch.zeros(1, 4, 4),
            "concat_cond": torch.full((1, 4, 4), 2.0),
            "attn_cond": torch.full((2, 4, 4), 4.0),
        },
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="chain",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert cond["concatenate"].shape == (2, 1, 4, 4)
    assert cond["attention"].shape == (2, 2, 4, 4)


def test_build_conditioning_batch_chain_falls_back_to_image() -> None:
    samples = [
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4)},
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4) * 2},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="chain",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert torch.allclose(cond["concatenate"], cond["attention"])


def test_build_conditioning_batch_attention_prefers_text_embeddings() -> None:
    samples = [
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4)},
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4)},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    text_embeddings = torch.randn(2, 77, 32)
    cond = _build_conditioning_batch(
        conditioning_mode="attention",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
        text_embeddings=text_embeddings,
    )
    assert cond is text_embeddings


def test_build_conditioning_batch_chain_uses_text_for_attention_when_missing() -> None:
    samples = [
        {"target": torch.zeros(1, 4, 4), "concat_cond": torch.ones(1, 4, 4)},
        {"target": torch.zeros(1, 4, 4), "concat_cond": torch.ones(1, 4, 4) * 2},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    text_embeddings = torch.randn(2, 77, 32)
    cond = _build_conditioning_batch(
        conditioning_mode="chain",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
        text_embeddings=text_embeddings,
    )
    assert isinstance(cond, dict)
    assert cond["attention"] is text_embeddings


def test_build_conditioning_batch_inpainting_uses_mask_and_target_as_original() -> None:
    samples = [
        {"target": torch.randn(1, 4, 4), "mask": torch.zeros(1, 4, 4)},
        {"target": torch.randn(1, 4, 4), "mask": torch.ones(1, 4, 4)},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="inpainting",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert cond["mask"].shape == (2, 1, 4, 4)
    assert torch.allclose(cond["original"], targets)


def test_resolve_conditioning_save_tensor_inpainting_prefers_mask() -> None:
    sample = {
        "image": torch.full((1, 4, 4), 9.0),
        "mask": torch.ones(1, 4, 4),
    }
    selected = _resolve_conditioning_save_tensor(sample, "inpainting")
    assert selected is not None
    assert torch.allclose(selected, sample["mask"])


def test_resolve_conditioning_save_tensor_chain_prefers_concat_then_attn_then_image() -> None:
    sample = {
        "image": torch.full((1, 4, 4), 1.0),
        "attn_cond": torch.full((1, 4, 4), 2.0),
        "concat_cond": torch.full((1, 4, 4), 3.0),
    }
    selected = _resolve_conditioning_save_tensor(sample, "chain")
    assert selected is not None
    assert torch.allclose(selected, sample["concat_cond"])


def test_run_decode_uses_inference_pipeline_when_no_scheduler_overrides(monkeypatch, tmp_path) -> None:
    cfg = {
        "training": {"conditioning": "none", "num_inference_steps": 4},
        "model": {"model_type": "diffusion", "conditioning": "none", "scheduler": {}},
        "sampling": {},
    }
    called = {"generate": 0}

    class _DummyPipe:
        def generate(self, inputs):
            called["generate"] += 1
            return torch.zeros(inputs.sample_shape)

    monkeypatch.setattr("pipelines.samplers.diffusion_like.load_run_config", lambda _p: cfg)
    monkeypatch.setattr("pipelines.samplers.diffusion_like.resolve_checkpoint", lambda *_args, **_kwargs: tmp_path / "ckpt.pt")
    monkeypatch.setattr("pipelines.samplers.diffusion_like.build_sampling_dataset", lambda *_args, **_kwargs: type("D", (), {"data": [{}], "target_key": "target", "conditioning_key": None})())
    monkeypatch.setattr("pipelines.samplers.diffusion_like.resolve_sample_indices", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr("pipelines.samplers.diffusion_like.resolve_output_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pipelines.samplers.diffusion_like.build_diffusion_model", lambda *_args, **_kwargs: torch.nn.Identity())
    monkeypatch.setattr("pipelines.samplers.diffusion_like.resolve_conditioning_mode", lambda *_args, **_kwargs: "none")
    monkeypatch.setattr("pipelines.samplers.diffusion_like._build_inference_pipeline", lambda **_kwargs: (_DummyPipe(), 4))
    monkeypatch.setattr(
        "pipelines.samplers.diffusion_like.progress_batches",
        lambda *_args, **_kwargs: [([0], [{"target": torch.zeros(1, 4, 4)}])],
    )

    _run_decode(
        ckpt_dir=tmp_path,
        model_type="diffusion",
        batch_size=1,
        save=False,
        num_samples=1,
        seed=0,
    )
    assert called["generate"] == 1
