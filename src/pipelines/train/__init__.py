"""
Deprecated compatibility re-exports for legacy training wrappers.

Canonical training lives in `src.training` and is dispatched by `train.py` or
`TRAINER_REGISTRY`.
"""

from __future__ import annotations

from importlib import import_module
import sys as _sys

from compat._deprecation import warn_deprecated

if __name__ == "src.pipelines.train" and "pipelines.train" in _sys.modules:
    _canonical = _sys.modules["pipelines.train"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)

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

_prefix = f"{__name__}."
_alt_prefix = "pipelines.train." if __name__ == "src.pipelines.train" else "src.pipelines.train."
for _mod_name, _mod in list(_sys.modules.items()):
    if _mod_name.startswith(_prefix):
        _alias = _alt_prefix + _mod_name[len(_prefix):]
        _sys.modules.setdefault(_alias, _mod)

del _prefix, _alt_prefix, _mod_name, _mod, _sys
