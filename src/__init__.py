"""
Deprecated compatibility shim for the old `src` package root.

Use `genlib` as the canonical import root.
"""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "Importing from 'src' is deprecated. Use 'genlib' as the package root.",
    DeprecationWarning,
    stacklevel=2,
)

_genlib = importlib.import_module("genlib")
__version__ = getattr(_genlib, "__version__", "1.0.0")
__all__ = list(getattr(_genlib, "__all__", []))


def __getattr__(name: str):
    return getattr(_genlib, name)
