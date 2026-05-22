from __future__ import annotations

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
