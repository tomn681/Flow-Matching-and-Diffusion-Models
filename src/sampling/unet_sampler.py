from __future__ import annotations

from pipelines.samplers.diffusion_like import _run_decode, _run_evaluate

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register("unet")
class UNetSampler(BaseSampler):
    """Sampler for supervised UNet checkpoints."""

    def decode(self) -> None:
        _run_decode(model_type="unet", **self._generative_decode_like_kwargs)

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        _run_evaluate(model_type="unet", **self._generative_decode_like_kwargs)

