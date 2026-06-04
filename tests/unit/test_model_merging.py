from __future__ import annotations

import torch
import torch.nn as nn

from models import merge_models


def test_merge_models_weighted_average_for_floating_parameters() -> None:
    model_a = nn.Linear(3, 2)
    model_b = nn.Linear(3, 2)
    with torch.no_grad():
        model_a.weight.fill_(2.0)
        model_a.bias.fill_(4.0)
        model_b.weight.fill_(10.0)
        model_b.bias.fill_(20.0)

    merged = merge_models(model_a, model_b, alpha=0.25)

    assert torch.allclose(merged["weight"], torch.full_like(model_a.weight, 8.0))
    assert torch.allclose(merged["bias"], torch.full_like(model_a.bias, 16.0))


def test_merge_models_chooses_discrete_buffers_from_nearest_endpoint() -> None:
    class _WithCounter(nn.Module):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.register_buffer("counter", torch.tensor(value, dtype=torch.long))

    model_a = _WithCounter(3)
    model_b = _WithCounter(9)

    merged_a = merge_models(model_a, model_b, alpha=0.75)
    merged_b = merge_models(model_a, model_b, alpha=0.25)

    assert int(merged_a["counter"]) == 3
    assert int(merged_b["counter"]) == 9


def test_merge_models_rejects_shape_mismatch() -> None:
    model_a = nn.Linear(3, 2)
    model_b = nn.Linear(4, 2)

    try:
        merge_models(model_a, model_b)
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected merge_models to reject shape mismatch.")


def test_merge_models_rejects_alpha_out_of_range() -> None:
    model_a = nn.Linear(3, 2)
    model_b = nn.Linear(3, 2)

    try:
        merge_models(model_a, model_b, alpha=1.5)
    except ValueError as exc:
        assert "alpha must be in [0, 1]" in str(exc)
    else:
        raise AssertionError("Expected merge_models to reject invalid alpha.")
