from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None


def safe_torch_load(path, *, map_location=None, weights_only: bool = True):
    if torch is None:
        raise RuntimeError("safe_torch_load requires PyTorch to be installed.")
    if not weights_only:
        return torch.load(path, map_location=map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = list(output_dir.glob("*_last.pt")) + list(output_dir.glob("*_best.pt")) + list(output_dir.glob("interrupt_last.pt"))
    if not candidates:
        candidates = list(output_dir.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_checkpoint(state: dict, path: Path) -> None:
    if torch is None:
        raise RuntimeError("save_checkpoint requires PyTorch to be installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def maybe_load_checkpoint(path: Path | str | None, prefix: str, model, optimizer=None, scheduler=None, scaler=None):
    if path is None:
        return 1, float("inf")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        logging.warning("%s checkpoint not found: %s", prefix, ckpt_path)
        return 1, float("inf")
    payload = safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(payload["scaler"])
    start_epoch = int(payload.get("epoch", 0)) + 1
    best = float(payload.get("best_metric", float("inf")))
    logging.info("Resumed %s from %s (epoch %d)", prefix, ckpt_path, start_epoch)
    return start_epoch, best
