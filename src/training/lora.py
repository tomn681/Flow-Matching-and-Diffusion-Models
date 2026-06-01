from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nn.modules import LoRALinear


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def wrap_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    if rank <= 0:
        raise ValueError("rank must be > 0")

    candidates: list[tuple[str, nn.Module]] = list(model.named_modules())
    matches: list[str] = []
    for name, module in candidates:
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(target) for target in target_modules):
            matches.append(name)
    if not matches:
        joined = ", ".join(target_modules)
        raise ValueError(f"No target Linear modules matched for LoRA injection. Targets: [{joined}]")

    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name in matches:
        module = dict(candidates)[name]
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
    return model


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    out: dict[str, Any] = {"__metadata__": {}}
    meta = out["__metadata__"]
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            meta[name] = {"rank": int(module.rank), "alpha": float(module.alpha)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("LoRA checkpoint must be a dict of tensors.")
    metadata = state.get("__metadata__")
    metadata = metadata if isinstance(metadata, dict) else {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        saved_meta = metadata.get(name)
        if isinstance(saved_meta, dict):
            saved_rank = int(saved_meta.get("rank", module.rank))
            saved_alpha = float(saved_meta.get("alpha", module.alpha))
            if saved_rank != module.rank or saved_alpha != module.alpha:
                raise ValueError(
                    f"LoRA metadata mismatch at '{name}': checkpoint rank={saved_rank}/alpha={saved_alpha}, "
                    f"model rank={module.rank}/alpha={module.alpha}."
                )
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in state:
            module.lora_A.data.copy_(torch.as_tensor(state[a_key], dtype=module.lora_A.dtype))
        if b_key in state:
            module.lora_B.data.copy_(torch.as_tensor(state[b_key], dtype=module.lora_B.dtype))


class LoRAWrapper:
    """Backwards-compatible namespace alias for LoRA helpers."""

    wrap = staticmethod(wrap_lora)
    save_lora_weights = staticmethod(save_lora_weights)
    load_lora_weights = staticmethod(load_lora_weights)


__all__ = ["LoRAWrapper", "wrap_lora", "save_lora_weights", "load_lora_weights"]
