from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Iterable, Mapping, cast


@dataclass(frozen=True)
class RegistryHub:
    model_families: Any | None = None


def discover_plugins(group: str = "genlib.plugins") -> list[str]:
    """
    Discover plugin entry points by group and return loaded plugin names.

    Entry points that fail to load are skipped so plugin discovery remains
    best-effort and does not break application startup.
    """
    loaded: list[str] = []
    entry_points = metadata.entry_points()

    if hasattr(entry_points, "select"):
        candidates = cast(Iterable[Any], entry_points.select(group=group))
    else:
        legacy_mapping = cast(Mapping[str, Iterable[Any]], entry_points)
        candidates = legacy_mapping.get(group, ())

    for ep in candidates:
        try:
            ep.load()
        except Exception as exc:
            warnings.warn(f"Failed to load plugin '{ep.name}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        loaded.append(ep.name)
    return loaded


def load_plugins(group: str = "genlib.plugins", *, hub: RegistryHub | None = None) -> list[str]:
    loaded: list[str] = []
    entry_points = metadata.entry_points()

    if hasattr(entry_points, "select"):
        candidates = cast(Iterable[Any], entry_points.select(group=group))
    else:
        legacy_mapping = cast(Mapping[str, Iterable[Any]], entry_points)
        candidates = legacy_mapping.get(group, ())

    for ep in candidates:
        try:
            register = ep.load()
            if callable(register):
                try:
                    signature = inspect.signature(register)
                except Exception:
                    signature = None
                if signature is not None and len(signature.parameters) > 0:
                    register(hub)
                else:
                    register()
        except Exception as exc:
            warnings.warn(f"Failed to load plugin '{ep.name}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        loaded.append(ep.name)
    return loaded


__all__ = ["RegistryHub", "discover_plugins", "load_plugins"]
