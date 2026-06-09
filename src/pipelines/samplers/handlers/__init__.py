"""
Deprecated compatibility re-exports for legacy sampler handlers.

Canonical runtime samplers live in `src.sampling` and are dispatched by
`run_model.py` or `SAMPLER_REGISTRY`.
"""

from __future__ import annotations

from importlib import import_module

from compat._deprecation import warn_deprecated

_EXPORTS = {
    "ModelHandler": ("compat.legacy_samplers", "ModelHandler"),
    "DiffusionHandler": ("compat.legacy_samplers", "DiffusionHandler"),
    "FlowMatchingHandler": ("compat.legacy_samplers", "FlowMatchingHandler"),
    "VAEHandler": ("compat.legacy_samplers", "VAEHandler"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        api=f"{__name__}.{name}",
        replacement="sampling.*Sampler classes, SAMPLER_REGISTRY, or python run_model.py",
        stacklevel=2,
    )
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)


__all__ = list(_EXPORTS.keys())
