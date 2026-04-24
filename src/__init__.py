"""
LDCT_v2 diffusion and flow-matching library.

This package exposes:
- `nn`: low-level neural network building blocks
- `models`: high-level model compositions
- `pipelines`: training / evaluation entrypoints
- `utils`: config, checkpoint, and dataset helpers

The dataset classes live under `src.datasets` but are not re-exported here.
Training can be launched either through the repo-root `train.py` wrapper or
through `python -m src.train` when explicit overrides are needed.
"""

from . import models, nn, pipelines, utils

__all__ = ["models", "nn", "utils", "pipelines"]

# Expose top-level aliases (nn, pipelines, models, utils) so imports can use
# `pipelines.train.vae` instead of `src.pipelines.train.vae`.
import sys as _sys
for _name in ("nn", "pipelines", "models", "utils"):
    _sys.modules.setdefault(_name, _sys.modules[f"{__name__}.{_name}"])
del _sys, _name
