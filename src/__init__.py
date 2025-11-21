"""
LDCT_v2 diffusion and flow-matching library.

This package currently exposes the low-level neural network building blocks
under `nn`, higher-level model compositions under `models`, dataset/utilities
under `utils`, and executable pipelines under `pipelines` (dispatched via
`python -m src.train` for training and `python -m src.sample` for sampling).
"""

from . import models, nn, pipelines, utils

__all__ = ["models", "nn", "utils", "pipelines"]

# Expose top-level aliases (nn, pipelines, models, utils) so imports can use
# `pipelines.train.vae` instead of `src.pipelines.train.vae`.
import sys as _sys
for _name in ("nn", "pipelines", "models", "utils"):
    _sys.modules.setdefault(_name, _sys.modules[f"{__name__}.{_name}"])
del _sys, _name
