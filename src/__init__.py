"""
LDCT_v2 diffusion and flow-matching library.

This package currently exposes the low-level neural network building blocks
under `nn`, higher-level model compositions under `models`, dataset/utilities
under `utils`, and executable pipelines under `pipelines` (dispatched via
`python -m src.train` for training and `python run_model.py` for sampling).
"""

from . import configs, core, datasets, losses, models, nn, noise, pipelines, scheduling, training, utils

__all__ = [
    "configs",
    "core",
    "datasets",
    "losses",
    "models",
    "nn",
    "noise",
    "pipelines",
    "scheduling",
    "training",
    "utils",
]

# Expose top-level aliases (nn, pipelines, models, utils) so imports can use
# `pipelines.train.vae` instead of `src.pipelines.train.vae`.
import sys as _sys
for _name in ("nn", "pipelines", "models", "utils", "core", "noise", "losses", "configs", "scheduling", "training", "datasets"):
    _sys.modules.setdefault(_name, _sys.modules[f"{__package__}.{_name}"])
del _sys, _name
