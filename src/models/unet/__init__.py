"""
UNet-based model definitions.

Re-exports the efficient N-Dimensional UNet implementation built from the core
neural network operators.
"""

import sys as _sys

if __name__ == "src.models.unet" and "models.unet" in _sys.modules:
    _canonical = _sys.modules["models.unet"]
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
    _alt_prefix = "models.unet." if __name__ == "src.models.unet" else "src.models.unet."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
