from __future__ import annotations

import torch

from models.adapters.weight_mappers import (
    load_hf_unet_weights,
    map_hf_unet_key_to_ours,
    map_hf_unet_to_ours,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        class _ConvWrap(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        class _AttnWrap(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = torch.nn.Linear(8, 8)
                self.to_k = torch.nn.Linear(8, 8)
                self.to_v = torch.nn.Linear(8, 8)
                self.to_out = torch.nn.Sequential(torch.nn.Linear(8, 8))

        self.conv1 = _ConvWrap()
        self.emb_layers = torch.nn.Linear(8, 8)
        self.attn = _AttnWrap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)


def _hf_like_state_from_model(model: _TinyModel) -> dict[str, torch.Tensor]:
    ours = model.state_dict()
    return {
        "conv1.weight": ours["conv1.conv.weight"],
        "conv1.bias": ours["conv1.conv.bias"],
        "time_emb_proj.weight": ours["emb_layers.weight"],
        "time_emb_proj.bias": ours["emb_layers.bias"],
        "attn.query.weight": ours["attn.to_q.weight"],
        "attn.query.bias": ours["attn.to_q.bias"],
        "attn.key.weight": ours["attn.to_k.weight"],
        "attn.key.bias": ours["attn.to_k.bias"],
        "attn.value.weight": ours["attn.to_v.weight"],
        "attn.value.bias": ours["attn.to_v.bias"],
        "attn.proj_attn.weight": ours["attn.to_out.0.weight"],
        "attn.proj_attn.bias": ours["attn.to_out.0.bias"],
    }


def test_map_hf_unet_key_to_ours_rewrites_expected_patterns() -> None:
    assert map_hf_unet_key_to_ours("attn.query.weight") == "attn.to_q.weight"
    assert map_hf_unet_key_to_ours("time_emb_proj.bias") == "emb_layers.bias"
    assert map_hf_unet_key_to_ours("attn.proj_attn.weight") == "attn.to_out.0.weight"


def test_map_hf_unet_to_ours_covers_all_keys_and_shapes() -> None:
    model = _TinyModel()
    hf_state = _hf_like_state_from_model(model)
    mapped = map_hf_unet_to_ours(hf_state, target_state_dict=model.state_dict())
    assert len(mapped) == len(hf_state)
    for key, tensor in mapped.items():
        assert key in model.state_dict()
        assert tuple(tensor.shape) == tuple(model.state_dict()[key].shape)


def test_map_hf_unet_to_ours_raises_on_shape_mismatch() -> None:
    model = _TinyModel()
    hf_state = _hf_like_state_from_model(model)
    hf_state["attn.query.weight"] = torch.randn(4, 4)
    try:
        map_hf_unet_to_ours(hf_state, target_state_dict=model.state_dict())
        assert False, "Expected shape mismatch error"
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)


def test_load_hf_unet_weights_smoke_and_roundtrip() -> None:
    model = _TinyModel()
    hf_state = _hf_like_state_from_model(model)
    load_hf_unet_weights(model, "dummy/model", hf_state_dict=hf_state)

    x = torch.randn(2, 4, 8, 8)
    y = model(x)
    assert torch.isfinite(y).all()

    state = model.state_dict()
    clone = _TinyModel()
    clone.load_state_dict(state)
    for k, v in clone.state_dict().items():
        assert torch.allclose(v, state[k])
