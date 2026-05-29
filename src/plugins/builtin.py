"""Built-in plugin entrypoint target used by `genlib.plugins` discovery."""

from __future__ import annotations


def register() -> str:
    """Return the built-in plugin identifier."""
    return "builtin"
