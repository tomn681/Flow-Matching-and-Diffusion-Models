from __future__ import annotations

import warnings
from importlib import metadata


def discover_plugins(group: str = "genlib.plugins") -> list[str]:
    """
    Discover plugin entry points by group and return loaded plugin names.

    Entry points that fail to load are skipped so plugin discovery remains
    best-effort and does not break application startup.
    """
    loaded: list[str] = []
    entry_points = metadata.entry_points()

    if hasattr(entry_points, "select"):
        candidates = entry_points.select(group=group)
    else:
        candidates = entry_points.get(group, ())

    for ep in candidates:
        try:
            ep.load()
        except Exception as exc:
            warnings.warn(f"Failed to load plugin '{ep.name}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        loaded.append(ep.name)
    return loaded


__all__ = ["discover_plugins"]
