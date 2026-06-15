from __future__ import annotations

import sys as _sys
from core.registry import Registry


LOSS_REGISTRY = Registry("losses")

_module = _sys.modules[__name__]
if __name__.startswith("genlib.losses."):
    _sys.modules.setdefault(__name__.replace("genlib.losses.", "losses.", 1), _module)
elif __name__.startswith("src.losses."):
    _sys.modules.setdefault(__name__.replace("src.losses.", "losses.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.losses.", "genlib.losses.", 1), _module)
elif __name__.startswith("losses."):
    _sys.modules.setdefault(__name__.replace("losses.", "src.losses.", 1), _module)
    _sys.modules.setdefault(__name__.replace("losses.", "genlib.losses.", 1), _module)
del _module, _sys
