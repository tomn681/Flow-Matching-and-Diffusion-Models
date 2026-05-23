from __future__ import annotations

import torch

from pipelines.samplers.diffusion_like import _build_conditioning_batch, _resolve_conditioning_save_tensor


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
