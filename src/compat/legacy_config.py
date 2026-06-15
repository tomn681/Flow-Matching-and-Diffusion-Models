from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

try:
    from ..configs.migration import normalize_aliases
except ImportError:  # top-level compat package import path
    from configs.migration import normalize_aliases


def adapt_legacy_config_v1(config: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt older v1-style loose configs into current framework shape.

    This adapter is intentionally conservative: it normalizes known aliases and
    ensures `training`/`model` sections exist so downstream code can validate
    consistently.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")

    adapted = cast(dict[str, Any], normalize_aliases(deepcopy(config)))
    adapted.setdefault("training", {})
    adapted.setdefault("model", {})
    if not isinstance(adapted["training"], dict):
        adapted["training"] = {}
    if not isinstance(adapted["model"], dict):
        adapted["model"] = {}
    return adapted


__all__ = ["adapt_legacy_config_v1"]
