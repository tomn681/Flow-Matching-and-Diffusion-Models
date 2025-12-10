from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None

__all__ = [
    "load_json_config",
    "save_json_config",
    "set_seed",
    "resolve_device",
    "summarize_model",
    "allocate_run_dir",
    "latest_checkpoint",
    "save_checkpoint",
]


def load_json_config(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def save_json_config(path: Path | str, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(cfg, fh, indent=2)


def allocate_run_dir(base: Path | str) -> Path:
    """
    Given a base directory, pick the next available run directory with suffix _runN.
    Example: base checkpoints/mnist -> checkpoints/mnist_run1, _run2, etc.
    """
    base = Path(base)
    parent = base.parent
    stem = base.name
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(stem)}_run(\d+)$")
    existing = []
    for entry in parent.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                existing.append(int(m.group(1)))
    next_id = (max(existing) + 1) if existing else 1
    return parent / f"{stem}_run{next_id}"


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(value, default: "torch.device") -> "torch.device":
    if torch is None:
        raise RuntimeError("resolve_device requires PyTorch to be installed.")
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "none":
        return default
    return value if isinstance(value, torch.device) else torch.device(value)


def summarize_model(model: "torch.nn.Module", vae_cfg: dict, training_cfg: dict) -> None:
    if torch is None:
        raise RuntimeError("summarize_model requires PyTorch to be installed.")

    show = training_cfg.get("show_model_summary", True)
    if not show:
        return

    def fmt(count: int) -> str:
        if count >= 1e6:
            return f"{count/1e6:.2f}M"
        if count >= 1e3:
            return f"{count/1e3:.2f}K"
        return str(count)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Build an input shape for summary
    spatial_dims = vae_cfg.get("spatial_dims", 2)
    in_ch = vae_cfg.get("in_channels", 3)
    res = vae_cfg.get("resolution", 256)
    if spatial_dims == 3:
        input_size = (1, in_ch, res, res, res)
    elif spatial_dims == 1:
        input_size = (1, in_ch, res)
    else:
        input_size = (1, in_ch, res, res)

    lines = []
    try:
        from torchinfo import summary  # type: ignore

        summary_obj = summary(
            model,
            input_size=input_size,
            col_names=("output_size", "num_params"),
            verbose=0,
            depth=10,
        )
        allowed = ("conv", "linear", "attention", "pool")
        header = f"{'idx':>4}  {'module':<40}  {'params':>10}  {'output':<20}"
        lines.append("Model summary (compact):")
        lines.append(header)
        lines.append("-" * len(header))
        for layer in summary_obj.summary_list:
            cls_name = layer.class_name.lower()
            if not any(k in cls_name for k in allowed):
                continue
            name = layer.layer_name
            params = layer.num_params
            out = str(layer.output_size)
            lines.append(f"{layer.depth:>4}  {name[:40]:<40}  {fmt(params):>10}  {out:<20}")
    except Exception as exc:
        logging.debug("torchinfo summary skipped (%s)", exc)
        header = f"{'idx':>4}  {'module':<40}  {'params':>10}"
        lines.append("Model summary (compact):")
        lines.append(header)
        lines.append("-" * len(header))
        for idx, (name, module) in enumerate(model.named_modules()):
            if name == "":
                continue
            params = sum(p.numel() for p in module.parameters())
            if params == 0:
                continue
            cls_lower = module.__class__.__name__.lower()
            if not any(k in cls_lower for k in ("conv", "linear", "attention", "pool")):
                continue
            lines.append(f"{idx:>4}  {name[:40]:<40}  {fmt(params):>10}")

    lines.append(f"Total params: {fmt(total_params)} ({total_params}) | Trainable: {fmt(trainable_params)} ({trainable_params})")

    for line in lines:
        logging.info(line)
        print(line, flush=True)


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = list(output_dir.glob("vae_last.pt")) + list(output_dir.glob("vae_best.pt"))
    if not candidates:
        candidates = list(output_dir.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_checkpoint(state: dict, path: Path) -> None:
    if torch is None:
        raise RuntimeError("save_checkpoint requires PyTorch to be installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
