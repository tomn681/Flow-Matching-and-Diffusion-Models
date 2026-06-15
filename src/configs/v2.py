from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any


_ALLOWED_TOP_LEVEL_KEYS = {
    "_base_",
    "__config_path__",
    "training",
    "model",
    "dataset",
    "sampling",
    "dataset_class",
    "data_root",
    "preprocess_kwargs",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Config at {path} must be a JSON object.")
    return raw


def _resolve_base_refs(config: dict[str, Any], *, config_path: Path | None) -> dict[str, Any]:
    base_spec = config.get("_base_")
    if not base_spec:
        return copy.deepcopy(config)
    if config_path is None:
        raise ValueError("Config uses _base_ but has no filesystem path for resolution.")
    refs = [base_spec] if isinstance(base_spec, (str, Path)) else list(base_spec)
    merged: dict[str, Any] = {}
    for ref in refs:
        base_path = Path(ref)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_cfg = resolve_config(base_path)
        merged = _deep_merge(merged, base_cfg)
    return _deep_merge(merged, config)


def _coerce_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def apply_dotted_overrides(config: dict[str, Any], overrides: dict[str, Any] | list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    normalized: dict[str, Any] = {}
    if isinstance(overrides, list):
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override '{item}' must use dotted.path=value syntax.")
            key, value = item.split("=", 1)
            normalized[key.strip()] = _coerce_override_value(value)
    else:
        normalized = dict(overrides)

    resolved = copy.deepcopy(config)
    for dotted_key, value in normalized.items():
        parts = [part.strip() for part in str(dotted_key).split(".") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid override key '{dotted_key}'.")
        target = resolved
        for part in parts[:-1]:
            child = target.get(part)
            if child is None:
                child = {}
                target[part] = child
            if not isinstance(child, dict):
                raise TypeError(f"Cannot apply override '{dotted_key}': '{part}' is not a mapping.")
            target = child
        target[parts[-1]] = value
    return resolved


def semantic_validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(key for key in config.keys() if key not in _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {', '.join(unknown)}")


def validate_runtime_semantics(config: dict[str, Any], *, steps_per_epoch: int | None = None) -> None:
    training = config.get("training", {}) if isinstance(config, dict) else {}
    epochs = int(training.get("epochs", 1) or 1)
    if int(training.get("gan_start", 0) or 0) > epochs:
        warnings.warn(
            f"training.gan_start={training.get('gan_start')} is beyond training.epochs={epochs}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if steps_per_epoch is not None and steps_per_epoch > 0:
        total_steps = epochs * int(steps_per_epoch)
        kl_anneal_steps = int(training.get("kl_anneal_steps", 0) or 0)
        if kl_anneal_steps > total_steps:
            warnings.warn(
                f"training.kl_anneal_steps={kl_anneal_steps} exceeds total training steps={total_steps}.",
                RuntimeWarning,
                stacklevel=2,
            )


def write_resolved_config_dump(output_dir: Path | str, config: dict[str, Any]) -> Path:
    output_path = Path(output_dir) / "resolved_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return output_path


def resolve_config(path_or_dict: Path | str | dict[str, Any], overrides: dict[str, Any] | list[str] | None = None) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        raw = copy.deepcopy(path_or_dict)
        config_path = Path(raw["__config_path__"]) if isinstance(raw.get("__config_path__"), str) else None
    else:
        config_path = Path(path_or_dict)
        raw = _load_raw_json(config_path)
    resolved = _resolve_base_refs(raw, config_path=config_path)
    resolved = apply_dotted_overrides(resolved, overrides)
    if config_path is not None:
        resolved["__config_path__"] = str(config_path)
    semantic_validate_config(resolved)
    return resolved


__all__ = [
    "apply_dotted_overrides",
    "resolve_config",
    "semantic_validate_config",
    "validate_runtime_semantics",
    "write_resolved_config_dump",
]
