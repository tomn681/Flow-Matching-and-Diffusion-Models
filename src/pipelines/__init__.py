"""
Training and inference pipelines for the diffusion / VAE components.

Training entrypoints live under `src.pipelines.train` and are dispatched via
`python train.py --config <json>`. Inference/visualisation helpers live under
`src.pipelines.samplers` and are dispatched via `python run_model.py --mode ...`.
"""

from __future__ import annotations

import importlib

__all__ = ["InferenceInputs", "InferencePipeline", "TextToImageInputs", "TextToImagePipeline"]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(".inference", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
