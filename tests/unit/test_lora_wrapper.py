from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nn.modules import LoRALinear
from training.lora import LoRAWrapper


class _TinyAttn(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        return self.to_out[0](q + k + v)


def test_lora_wrap_replaces_target_linears() -> None:
    model = _TinyAttn()
    LoRAWrapper.wrap(model, rank=4, alpha=1.0)
    assert isinstance(model.to_q, LoRALinear)
    assert isinstance(model.to_k, LoRALinear)
    assert isinstance(model.to_v, LoRALinear)
    assert isinstance(model.to_out[0], LoRALinear)


def test_lora_wrap_only_lora_params_require_grad() -> None:
    model = _TinyAttn()
    LoRAWrapper.wrap(model, rank=4, alpha=1.0)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(("lora_A" in name or "lora_B" in name) for name in trainable)


def test_lora_forward_shape_matches_base() -> None:
    model = _TinyAttn()
    x = torch.randn(2, 5, 8)
    y_base = model(x)

    LoRAWrapper.wrap(model, rank=4, alpha=1.0)
    y_lora = model(x)

    assert y_base.shape == y_lora.shape


def test_lora_save_load_round_trip(tmp_path: Path) -> None:
    model = _TinyAttn()
    LoRAWrapper.wrap(model, rank=4, alpha=1.0)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.fill_(0.25)
            module.lora_B.data.fill_(0.5)

    lora_path = tmp_path / "lora_only.pt"
    LoRAWrapper.save_lora_weights(model, lora_path)

    reloaded = _TinyAttn()
    LoRAWrapper.wrap(reloaded, rank=4, alpha=1.0)
    LoRAWrapper.load_lora_weights(reloaded, lora_path)

    for (n1, m1), (n2, m2) in zip(model.named_modules(), reloaded.named_modules()):
        if isinstance(m1, LoRALinear):
            assert n1 == n2
            assert torch.allclose(m1.lora_A, m2.lora_A)
            assert torch.allclose(m1.lora_B, m2.lora_B)


def test_lora_weights_file_smaller_than_full_state(tmp_path: Path) -> None:
    model = _TinyAttn()
    LoRAWrapper.wrap(model, rank=4, alpha=1.0)

    lora_path = tmp_path / "lora_only.pt"
    full_path = tmp_path / "full_state.pt"
    LoRAWrapper.save_lora_weights(model, lora_path)
    torch.save(model.state_dict(), full_path)

    assert lora_path.stat().st_size < full_path.stat().st_size

