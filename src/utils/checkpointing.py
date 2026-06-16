from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from safetensors.torch import load_file as _load_safetensors_file
    from safetensors.torch import save_file as _save_safetensors_file
except ImportError:  # pragma: no cover - optional at import time
    _load_safetensors_file = None
    _save_safetensors_file = None

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None

CHECKPOINT_FORMAT_VERSION = 1


def _checkpoint_weights_path(path: Path) -> Path:
    return path.with_suffix(".safetensors")


def _migrate_checkpoint_payload(payload: dict, *, path: Path | None = None) -> dict:
    migrated = dict(payload)
    format_version = int(migrated.get("format_version", 0) or 0)
    if format_version <= 0:
        if "current_epoch" in migrated and "epoch" not in migrated:
            migrated["epoch"] = migrated["current_epoch"]
        if "last_epoch" in migrated and "epoch" not in migrated:
            migrated["epoch"] = migrated["last_epoch"]
        migrated.setdefault("extra", {})
        migrated.setdefault("metadata", {})
        migrated["format_version"] = CHECKPOINT_FORMAT_VERSION
    if "weights_path" in migrated and path is not None:
        weights_path = Path(migrated["weights_path"])
        if not weights_path.is_absolute():
            migrated["weights_path"] = str((path.parent / weights_path).resolve())
    return migrated


def _split_checkpoint_state(state: dict, *, path: Path) -> tuple[dict, dict] | None:
    if "model" not in state or not isinstance(state.get("model"), dict):
        return None
    sidecar = dict(state)
    model_state = dict(sidecar["model"])
    sidecar["format_version"] = CHECKPOINT_FORMAT_VERSION
    sidecar.setdefault("metadata", {})
    sidecar["weights_path"] = _checkpoint_weights_path(path).name
    return sidecar, model_state


def safe_torch_load(path, *, map_location=None, weights_only: bool = True):
    if torch is None:
        raise RuntimeError("safe_torch_load requires PyTorch to be installed.")
    if not weights_only:
        return torch.load(path, map_location=map_location)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        raise RuntimeError(
            "This PyTorch build does not support torch.load(..., weights_only=True). "
            "Upgrade PyTorch to load framework checkpoints safely."
        )
    resolved_path = Path(path)
    if isinstance(payload, dict):
        payload = _migrate_checkpoint_payload(payload, path=resolved_path)
        weights_path = payload.get("weights_path")
        if weights_path is not None and "model" not in payload:
            if _load_safetensors_file is None:
                raise RuntimeError(
                    "Loading versioned framework checkpoints requires the `safetensors` package."
                )
            state_path = Path(weights_path)
            if not state_path.is_absolute():
                state_path = resolved_path.parent / state_path
            payload["model"] = _load_safetensors_file(str(state_path), device=str(map_location or "cpu"))
    return payload


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
    tmp_weights: Path | None = None
    try:
        split_state = _split_checkpoint_state(state, path=path)
        if split_state is not None:
            if _save_safetensors_file is None:
                raise RuntimeError(
                    "Saving versioned framework checkpoints requires the `safetensors` package."
                )
            sidecar, model_state = split_state
            weights_path = _checkpoint_weights_path(path)
            tmp_weights = weights_path.with_name(f"{weights_path.name}.{os.getpid()}.tmp")
            _save_safetensors_file(model_state, str(tmp_weights))
            torch.save(sidecar, tmp_path)
            os.replace(tmp_weights, weights_path)
        else:
            torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_weights is not None and tmp_weights.exists():
            tmp_weights.unlink(missing_ok=True)
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
