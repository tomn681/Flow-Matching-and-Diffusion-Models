"""
Training and inference pipelines for the diffusion / VAE components.

Training entrypoints live under `src.pipelines.train` and are dispatched via
`python train.py --config <json>`. Inference/visualisation helpers live under
`src.pipelines.samplers` and are dispatched via `python run_model.py --mode ...`.
"""

from __future__ import annotations

import importlib
import sys as _sys

if __name__ == "src.pipelines" and "pipelines" in _sys.modules:
    _canonical = _sys.modules["pipelines"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    __all__ = ["InferenceInputs", "InferencePipeline", "TextToImageInputs", "TextToImagePipeline"]

    def __getattr__(name: str):
        if name in __all__:
            module = importlib.import_module(".inference", __name__)
            return getattr(module, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _prefix = f"{__name__}."
    _alt_prefix = "pipelines." if __name__ == "src.pipelines" else "src.pipelines."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
