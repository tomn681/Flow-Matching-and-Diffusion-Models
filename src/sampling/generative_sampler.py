from __future__ import annotations

from pathlib import Path

import torch

from pipelines.samplers.diffusion_like import _run_debug_compare, _run_decode, _run_encode, _run_evaluate
from noise.reflow import generate_reflow_pairs as _generate_reflow_pairs_impl
from utils.model_utils.diffusion_utils import build_diffusion_model
from utils.sampling_utils import load_run_config
from pipelines.utils import build_scheduler

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

    def generate_reflow_pairs(self) -> None:
        cfg = load_run_config(self.ckpt_dir)
        training_cfg = cfg["training"]
        model_cfg = cfg["model"]

        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device) if self.device else default_device

        model = build_diffusion_model(
            cfg,
            device,
            ckpt_path=str(self.ckpt_dir),
            model_type=self.model_type,
            set_eval=True,
        )
        scheduler, inferred_steps = build_scheduler(model_cfg.get("scheduler", {}), training_cfg)

        resolution = int(model_cfg.get("resolution", 256))
        channels = int(model_cfg.get("in_channels", 1))
        spatial_dims = int(model_cfg.get("spatial_dims", 2))
        sample_shape = (channels, *([resolution] * spatial_dims))

        num_pairs = self.num_pairs
        if num_pairs is None:
            num_pairs = int(model_cfg.get("num_pairs", 50000))

        output_dir = Path(self.output_dir) if self.output_dir else self.ckpt_dir / "reflow_pairs"

        _generate_reflow_pairs_impl(
            model=model,
            scheduler=scheduler,
            num_pairs=int(num_pairs),
            sample_shape=tuple(sample_shape),
            device=device,
            output_dir=output_dir,
            num_inference_steps=int(self.num_inference_steps or inferred_steps),
            batch_size=int(self.batch_size),
        )
        print(f"Generated {int(num_pairs)} reflow pairs in {output_dir}")


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


@SAMPLER_REGISTRY.register("reflow")
class ReflowSampler(GenerativeSampler):
    model_type = "reflow"
