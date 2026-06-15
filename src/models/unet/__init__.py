"""
UNet-based model definitions.

Re-exports the efficient N-Dimensional UNet implementation built from the core
neural network operators.
"""

import sys as _sys

_canonical_name = None
if __name__ in {"src.models.unet", "genlib.models.unet"} and "models.unet" in _sys.modules:
    _canonical_name = "models.unet"
elif __name__ == "models.unet" and "src.models.unet" in _sys.modules:
    _canonical_name = "src.models.unet"
elif __name__ == "models.unet" and "genlib.models.unet" in _sys.modules:
    _canonical_name = "genlib.models.unet"

if _canonical_name is not None:
    _canonical = _sys.modules[_canonical_name]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .base import BaseUNetND
    from .condition import UNet2DConditionND
    from .diffusers import UNetDiffusersND, UNetExactND
    from .efficient import EfficientUNetND, TimestepEmbedSequential
    from .video import VideoUNetND

    __all__ = [
        "BaseUNetND",
        "EfficientUNetND",
        "TimestepEmbedSequential",
        "UNetDiffusersND",
        "UNetExactND",
        "UNet2DConditionND",
        "VideoUNetND",
    ]

    _prefix = f"{__name__}."
    for _mod_name, _mod in list(_sys.modules.items()):
        if not _mod_name.startswith(_prefix):
            continue
        _suffix = _mod_name[len(_prefix):]
        for _alias_prefix in ("models.unet.", "src.models.unet.", "genlib.models.unet."):
            _sys.modules.setdefault(_alias_prefix + _suffix, _mod)

del _sys
