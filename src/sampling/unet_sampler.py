from __future__ import annotations

import sys as _sys

from pipelines.samplers.diffusion_like import _run_decode, _run_evaluate

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register("unet")
class UNetSampler(BaseSampler):
    """Sampler for supervised UNet checkpoints."""

    supported_modes = frozenset({"build_tensor_cache", "decode", "sample", "evaluate"})

    def decode(self) -> None:
        _run_decode(model_type="unet", **self._generative_decode_like_kwargs)

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        _run_evaluate(model_type="unet", **self._generative_decode_like_kwargs)


_module = _sys.modules[__name__]
if __name__.startswith("genlib.sampling."):
    _sys.modules.setdefault(__name__.replace("genlib.sampling.", "sampling.", 1), _module)
elif __name__.startswith("src.sampling."):
    _sys.modules.setdefault(__name__.replace("src.sampling.", "sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.sampling.", "genlib.sampling.", 1), _module)
elif __name__.startswith("sampling."):
    _sys.modules.setdefault(__name__.replace("sampling.", "src.sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("sampling.", "genlib.sampling.", 1), _module)
del _module, _sys
