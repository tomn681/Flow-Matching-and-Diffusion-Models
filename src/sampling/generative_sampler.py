from __future__ import annotations

from pipelines.samplers.diffusion_like import _run_debug_compare, _run_decode, _run_encode, _run_evaluate

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


class GenerativeSampler(BaseSampler):
    model_type: str

    def encode(self) -> None:
        _run_encode(model_type=self.model_type, timestep=self.timestep, **self._common_kwargs)

    def decode(self) -> None:
        _run_decode(model_type=self.model_type, **self._generative_decode_like_kwargs)

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        _run_evaluate(model_type=self.model_type, **self._generative_decode_like_kwargs)

    def debug_compare(self) -> None:
        _run_debug_compare(model_type=self.model_type, **self._generative_debug_compare_kwargs)


@SAMPLER_REGISTRY.register("diffusion")
class DiffusionSampler(GenerativeSampler):
    model_type = "diffusion"


@SAMPLER_REGISTRY.register("flow_matching")
class FlowMatchingSampler(GenerativeSampler):
    model_type = "flow_matching"


@SAMPLER_REGISTRY.register("consistency")
class ConsistencySampler(GenerativeSampler):
    model_type = "consistency"


@SAMPLER_REGISTRY.register("edm")
class EDMSampler(GenerativeSampler):
    model_type = "edm"


@SAMPLER_REGISTRY.register("rectified_flow")
class RectifiedFlowSampler(GenerativeSampler):
    model_type = "rectified_flow"
