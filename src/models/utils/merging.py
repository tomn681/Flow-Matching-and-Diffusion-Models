from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def merge_models(model_a: nn.Module, model_b: nn.Module, alpha: float = 0.5) -> OrderedDict[str, torch.Tensor]:
    """
    Return a merged state dict via weighted averaging.

    Floating-point and complex tensors are interpolated as:
    `alpha * a + (1 - alpha) * b`.

    Non-floating tensors (for example integer counters or boolean masks) are
    copied from the nearest endpoint because interpolation would be undefined.
    For `alpha >= 0.5`, values come from `model_a`; otherwise they come from
    `model_b`.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        details: list[str] = []
        if only_a:
            details.append(f"only in model_a: {only_a}")
        if only_b:
            details.append(f"only in model_b: {only_b}")
        raise ValueError("Model state dict keys do not match; " + "; ".join(details))

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    choose_a_for_discrete = alpha >= 0.5

    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}."
            )
        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"Dtype mismatch for key '{key}': {tensor_a.dtype} vs {tensor_b.dtype}."
            )

        if torch.is_floating_point(tensor_a) or torch.is_complex(tensor_a):
            merged[key] = alpha * tensor_a + (1.0 - alpha) * tensor_b
        else:
            source = tensor_a if choose_a_for_discrete else tensor_b
            merged[key] = source.clone()

    return merged


__all__ = ["merge_models"]
