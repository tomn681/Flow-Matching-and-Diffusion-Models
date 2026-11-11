from __future__ import annotations

import logging
import sys as _sys

from pipelines.samplers import autoencoder_like as autoencoder_sampler

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register("vae")
class VAESampler(BaseSampler):
    supported_modes = frozenset({"build_tensor_cache", "debug_compare", "encode", "decode", "sample", "evaluate"})

    @property
    def _vae_decode_kwargs(self) -> dict:
        kwargs = dict(self._common_kwargs)
        kwargs["save_input"] = self.save_input
        kwargs["save_conditioning"] = self.save_conditioning
        return kwargs

    def _warn_ignored_iterative_options(self) -> None:
        ignored: list[str] = []
        if self.timestep is not None:
            ignored.append(f"timestep={self.timestep}")
        if self.scheduler is not None:
            ignored.append(f"scheduler={self.scheduler!r}")
        if self.num_inference_steps is not None:
            ignored.append(f"num_inference_steps={self.num_inference_steps}")
        if self.start_step is not None:
            ignored.append(f"start_step={self.start_step}")
        if self.last_n_steps is not None:
            ignored.append(f"last_n_steps={self.last_n_steps}")
        if self.cfg_rescale != 0.0:
            ignored.append(f"cfg_rescale={self.cfg_rescale}")
        if ignored:
            logging.warning(
                "VAE runtime is a single encode/decode pass; ignoring iterative sampling options: %s.",
                ", ".join(ignored),
            )

    def encode(self) -> None:
        self._warn_ignored_iterative_options()
        autoencoder_sampler.encode(**self._common_kwargs, timestep=self.timestep)

    def decode(self) -> None:
        self._warn_ignored_iterative_options()
        autoencoder_sampler.decode(**self._vae_decode_kwargs)

    def sample(self) -> None:
        self._warn_ignored_iterative_options()
        autoencoder_sampler.sample(**self._decode_like_kwargs)

    def evaluate(self) -> None:
        self._warn_ignored_iterative_options()
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
