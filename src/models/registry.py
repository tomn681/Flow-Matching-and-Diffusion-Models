from __future__ import annotations

import sys as _sys
import torch.nn as nn

from core.registry import Registry


MODEL_REGISTRY = Registry[nn.Module]("models", base_type=nn.Module)


__all__ = ["MODEL_REGISTRY"]

_module = _sys.modules[__name__]
if __name__.startswith("genlib.models."):
    _sys.modules.setdefault(__name__.replace("genlib.models.", "models.", 1), _module)
elif __name__.startswith("src.models."):
    _sys.modules.setdefault(__name__.replace("src.models.", "models.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.models.", "genlib.models.", 1), _module)
elif __name__.startswith("models."):
    _sys.modules.setdefault(__name__.replace("models.", "src.models.", 1), _module)
    _sys.modules.setdefault(__name__.replace("models.", "genlib.models.", 1), _module)
del _module, _sys
