"""
Deprecated compatibility re-exports for legacy training wrappers.

Canonical training lives in `src.training` and is dispatched by `train.py` or
`TRAINER_REGISTRY`.
"""

from __future__ import annotations

from importlib import import_module

from compat._deprecation import warn_deprecated

_EXPORTS = {
    "train_diffusion": ("compat.legacy_training", "train_diffusion"),
    "train_flow_matching": ("compat.legacy_training", "train_flow_matching"),
    "train_vae": ("compat.legacy_training", "train_vae"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        api=f"{__name__}.{name}",
        replacement="training.TRAINER_REGISTRY[...] or python train.py --config ...",
        stacklevel=2,
    )
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)


__all__ = list(_EXPORTS.keys())
