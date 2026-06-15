from __future__ import annotations

import sys as _sys

from pipelines.samplers import autoencoder_like as autoencoder_sampler

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register("vae")
class VAESampler(BaseSampler):
    def encode(self) -> None:
        autoencoder_sampler.encode(**self._common_kwargs, timestep=self.timestep)

    def decode(self) -> None:
        autoencoder_sampler.decode(**self._decode_like_kwargs)

    def sample(self) -> None:
        autoencoder_sampler.sample(**self._decode_like_kwargs)

    def evaluate(self) -> None:
        autoencoder_sampler.evaluate(**self._decode_like_kwargs)

    def debug_compare(self) -> None:
        autoencoder_sampler.debug_compare(**self._debug_compare_kwargs)


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
