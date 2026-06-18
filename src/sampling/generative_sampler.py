from __future__ import annotations

import logging
from pathlib import Path
import sys as _sys

import torch

from core.noise_contracts import noise_family_for_model_type, warn_if_legacy_family_alias
from pipelines.samplers.diffusion_like import _run_debug_compare, _run_decode, _run_encode, _run_evaluate
from noise.reflow import generate_reflow_pairs as _generate_reflow_pairs_impl
from utils.model_utils.diffusion_utils import build_diffusion_model
from utils.sampling_utils import load_run_config, resolve_checkpoint
from pipelines.utils import build_scheduler

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


class GenerativeSampler(BaseSampler):
    model_type: str
    supported_modes = frozenset({"build_tensor_cache", "debug_compare", "encode", "decode", "sample", "evaluate"})

    def _warn_legacy_alias(self) -> None:
        warn_if_legacy_family_alias(self.model_type)

    def encode(self) -> None:
        self._warn_legacy_alias()
        _run_encode(model_type=self.model_type, timestep=self.timestep, **self._common_kwargs)

    def decode(self) -> None:
        self._warn_legacy_alias()
        _run_decode(model_type=self.model_type, **self._generative_decode_like_kwargs)

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        self._warn_legacy_alias()
        _run_evaluate(model_type=self.model_type, **self._generative_decode_like_kwargs)

    def debug_compare(self) -> None:
        self._warn_legacy_alias()
        _run_debug_compare(model_type=self.model_type, **self._generative_debug_compare_kwargs)

    def _generate_reflow_pairs(self) -> None:
        cfg = load_run_config(self.ckpt_dir)
        training_cfg = cfg["training"]
        model_cfg = cfg["model"]
        unet_cfg = model_cfg.get("unet", {})

        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device) if self.device else default_device
        ckpt_path = resolve_checkpoint(self.ckpt_dir, self.model_type)

        model = build_diffusion_model(
            cfg,
            device,
            ckpt_path=str(ckpt_path),
            set_eval=True,
            use_ema=self.use_ema,
        )
        scheduler, inferred_steps = build_scheduler(
            model_cfg.get("scheduler", {}),
            training_cfg,
            noise_family=noise_family_for_model_type(self.model_type),
        )

        resolution = int(
            unet_cfg.get("sample_size", model_cfg.get("resolution", training_cfg.get("img_size", 256)))
        )
        out_channels = int(unet_cfg.get("out_channels", model_cfg.get("out_channels", 1)))
        spatial_dims = int(unet_cfg.get("spatial_dims", model_cfg.get("spatial_dims", 2)))
        sample_shape = (out_channels, *([resolution] * spatial_dims))

        conditioning_mode = str(
            training_cfg.get("conditioning", model_cfg.get("conditioning", "none")) or "none"
        ).strip().lower()

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
            conditioning_mode=conditioning_mode,
        )
        logging.info("Generated %d reflow pairs in %s", int(num_pairs), output_dir)


@SAMPLER_REGISTRY.register("diffusion")
class DiffusionSampler(GenerativeSampler):
    model_type = "diffusion"


@SAMPLER_REGISTRY.register("video_unet")
class VideoUNetSampler(GenerativeSampler):
    """Sampler for video UNet checkpoints using the standard denoising loop.

    Video UNets currently share the same runtime denoiser loop as pixel diffusion
    models. Flow-matching-trained video checkpoints should grow a dedicated
    sampler class if/when that training path is introduced.
    """

    model_type = "diffusion"


@SAMPLER_REGISTRY.register("distillation")
class DistillationSampler(GenerativeSampler):
    """Sampler for distilled denoisers.

    Use `--num_inference_steps` to realize the student step-budget reduction at sampling time.
    """

    model_type = "diffusion"


@SAMPLER_REGISTRY.register("flow_matching")
class FlowMatchingSampler(GenerativeSampler):
    model_type = "flow_matching"
    supported_modes = GenerativeSampler.supported_modes | frozenset({"generate_reflow_pairs"})

    def generate_reflow_pairs(self) -> None:
        self._generate_reflow_pairs()


@SAMPLER_REGISTRY.register("consistency")
class ConsistencySampler(GenerativeSampler):
    model_type = "consistency"


@SAMPLER_REGISTRY.register("x0_denoising")
class X0DenoisingSampler(GenerativeSampler):
    model_type = "x0_denoising"


@SAMPLER_REGISTRY.register("edm")
class EDMSampler(GenerativeSampler):
    model_type = "edm"


@SAMPLER_REGISTRY.register("rectified_flow")
class RectifiedFlowSampler(GenerativeSampler):
    model_type = "rectified_flow"
    supported_modes = GenerativeSampler.supported_modes | frozenset({"generate_reflow_pairs"})

    def generate_reflow_pairs(self) -> None:
        self._generate_reflow_pairs()


@SAMPLER_REGISTRY.register("reflow")
class ReflowSampler(GenerativeSampler):
    model_type = "reflow"
    supported_modes = GenerativeSampler.supported_modes | frozenset({"generate_reflow_pairs"})

    def generate_reflow_pairs(self) -> None:
        self._generate_reflow_pairs()


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
