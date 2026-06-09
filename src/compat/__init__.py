"""
Compatibility wrappers for legacy training, sampling, and config APIs.

This package is intentionally non-canonical. New code should prefer the
registry-driven surfaces in `src.training`, `src.sampling`, and the root CLIs.
"""

from __future__ import annotations

from importlib import import_module

from .legacy_config import adapt_legacy_config_v1
from ._deprecation import warn_deprecated

_EXPORTS = {
    "ModelHandler": ("compat.legacy_samplers", "ModelHandler"),
    "DiffusionHandler": ("compat.legacy_samplers", "DiffusionHandler"),
    "FlowMatchingHandler": ("compat.legacy_samplers", "FlowMatchingHandler"),
    "VAEHandler": ("compat.legacy_samplers", "VAEHandler"),
    "train_diffusion": ("compat.legacy_training", "train_diffusion"),
    "train_flow_matching": ("compat.legacy_training", "train_flow_matching"),
    "train_vae": ("compat.legacy_training", "train_vae"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        api=f"{__name__}.{name}",
        replacement="canonical training/sampling registries and root CLIs",
        stacklevel=2,
    )
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)


__all__ = ["adapt_legacy_config_v1", *_EXPORTS.keys()]
