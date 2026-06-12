"""
Generative Modeling Framework
=============================

Curated public API across model definitions, training orchestration,
noise/scheduler abstractions, and compatibility wrappers.
"""

__version__ = "0.9.0"

import importlib as _importlib
import sys as _sys


def _bind_package(name: str):
    existing = _sys.modules.get(name)
    if existing is not None:
        _sys.modules[f"{__name__}.{name}"] = existing
        globals()[name] = existing
        return existing
    module = _importlib.import_module(f".{name}", __name__)
    _sys.modules.setdefault(name, module)
    globals()[name] = module
    return module


for _pkg_name in (
    "compat",
    "configs",
    "core",
    "datasets",
    "losses",
    "models",
    "nn",
    "noise",
    "pipelines",
    "plugins",
    "scheduling",
    "training",
    "utils",
):
    _bind_package(_pkg_name)

del _pkg_name

from .configs import FrameworkConfig, TrainingConfig, load_and_validate, load_config, validate_config
from .core.types import ModelOutput, NoisyBatch
from .losses import BaseLossComponent, LOSS_REGISTRY, LossAssembler
from .models import AutoencoderKL, BaseAutoencoder, BaseVAE, ControlNetND, DiTND, MODEL_REGISTRY, ModelFactory, VQVAE, merge_models
from .models.unet import BaseUNetND, EfficientUNetND, UNet2DConditionND, UNetDiffusersND
from .noise import ConsistencyNoise, DDPMNoise, EDMNoise, FlowMatchingNoise, NOISE_REGISTRY, RectifiedFlowNoise, ReflowNoise, X0DenoisingNoise, generate_reflow_pairs
from .pipelines import InferenceInputs, InferencePipeline, TextToImageInputs, TextToImagePipeline
from .sampling import (
    BaseSampler,
    ConsistencySampler,
    ControlNetSampler,
    DiffusionSampler,
    DistillationSampler,
    EDMSampler,
    FlowMatchingSampler,
    LatentDiffusionSampler,
    LatentFlowMatchingSampler,
    LatentRectifiedFlowSampler,
    ReflowSampler,
    RectifiedFlowSampler,
    SAMPLER_REGISTRY,
    UNetSampler,
    VAESampler,
    VideoUNetSampler,
)
from .scheduling import (
    CONDITIONING_ADAPTER_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    SCHEDULER_REGISTRY,
    TextConditioningAdapter,
    build_lr_scheduler,
    build_scheduler,
    build_text_conditioning_adapter,
    resolve_conditioning_adapter,
    resolve_conditioning_mode,
    resolve_scheduler_override,
    sample_with_scheduler,
)
from .training import (
    BaseTrainer,
    CheckpointCallback,
    ConsistencyTrainer,
    ControlNetTrainer,
    DiffusionTrainer,
    DistillationTrainer,
    EDMTrainer,
    FlowMatchingTrainer,
    GANTrainer,
    GenerativeTrainer,
    LatentDiffusionTrainer,
    LatentFlowMatchingTrainer,
    LatentGenerativeTrainer,
    LatentRectifiedFlowTrainer,
    LoRAWrapper,
    MetricsCSVCallback,
    ReflowTrainer,
    StepwiseResolutionSchedule,
    build_resolution_schedule,
    TRAINER_REGISTRY,
    VAETrainer,
    VisualizationCallback,
    RectifiedFlowTrainer,
    UNetTrainer,
)

__all__ = [
    "__version__",
    # Package modules
    "compat",
    "configs",
    "core",
    "datasets",
    "losses",
    "models",
    "nn",
    "noise",
    "pipelines",
    "plugins",
    "sampling",
    "scheduling",
    "training",
    "utils",
    # Core types
    "ModelOutput",
    "NoisyBatch",
    # Models
    "BaseAutoencoder",
    "BaseVAE",
    "AutoencoderKL",
    "VQVAE",
    "BaseUNetND",
    "EfficientUNetND",
    "UNetDiffusersND",
    "UNet2DConditionND",
    "ModelFactory",
    "merge_models",
    "ControlNetND",
    "DiTND",
    "InferenceInputs",
    "InferencePipeline",
    "TextToImageInputs",
    "TextToImagePipeline",
    # Training
    "BaseTrainer",
    "ControlNetTrainer",
    "GenerativeTrainer",
    "DiffusionTrainer",
    "DistillationTrainer",
    "FlowMatchingTrainer",
    "ConsistencyTrainer",
    "EDMTrainer",
    "RectifiedFlowTrainer",
    "ReflowTrainer",
    "StepwiseResolutionSchedule",
    "build_resolution_schedule",
    "LatentGenerativeTrainer",
    "LatentDiffusionTrainer",
    "LatentFlowMatchingTrainer",
    "LatentRectifiedFlowTrainer",
    "LoRAWrapper",
    "GANTrainer",
    "UNetTrainer",
    "VAETrainer",
    "CheckpointCallback",
    "MetricsCSVCallback",
    "VisualizationCallback",
    # Noise / scheduling / sampling
    "DDPMNoise",
    "FlowMatchingNoise",
    "ConsistencyNoise",
    "X0DenoisingNoise",
    "EDMNoise",
    "RectifiedFlowNoise",
    "ReflowNoise",
    "generate_reflow_pairs",
    "build_scheduler",
    "build_lr_scheduler",
    "build_text_conditioning_adapter",
    "TextConditioningAdapter",
    "resolve_conditioning_mode",
    "resolve_conditioning_adapter",
    "resolve_scheduler_override",
    "sample_with_scheduler",
    "BaseSampler",
    "ControlNetSampler",
    "VAESampler",
    "DiffusionSampler",
    "DistillationSampler",
    "FlowMatchingSampler",
    "LatentDiffusionSampler",
    "LatentFlowMatchingSampler",
    "LatentRectifiedFlowSampler",
    "ConsistencySampler",
    "EDMSampler",
    "RectifiedFlowSampler",
    "ReflowSampler",
    "VideoUNetSampler",
    "UNetSampler",
    # Config
    "FrameworkConfig",
    "TrainingConfig",
    "load_config",
    "load_and_validate",
    "validate_config",
    # Registries
    "MODEL_REGISTRY",
    "NOISE_REGISTRY",
    "LOSS_REGISTRY",
    "SCHEDULER_REGISTRY",
    "CONDITIONING_ADAPTER_REGISTRY",
    "LR_SCHEDULER_REGISTRY",
    "TRAINER_REGISTRY",
    "SAMPLER_REGISTRY",
    # Loss composition
    "BaseLossComponent",
    "LossAssembler",
]

# Expose top-level aliases (nn, pipelines, models, utils) so imports can use
# `pipelines.train.vae` instead of `src.pipelines.train.vae`.
_ROOT_ALIASES = ("nn", "pipelines", "models", "utils", "core", "noise", "losses", "configs", "scheduling", "training", "datasets", "compat", "plugins")
for _name in _ROOT_ALIASES:
    if _name in _sys.modules:
        _sys.modules[f"{__package__}.{_name}"] = _sys.modules[_name]
    else:
        _sys.modules[_name] = _sys.modules[f"{__package__}.{_name}"]

for _mod_name, _mod in list(_sys.modules.items()):
    for _root in _ROOT_ALIASES:
        _src_prefix = f"{__package__}.{_root}."
        _top_prefix = f"{_root}."
        if _mod_name.startswith(_src_prefix):
            _sys.modules.setdefault(_top_prefix + _mod_name[len(_src_prefix):], _mod)
        elif _mod_name.startswith(_top_prefix):
            _sys.modules.setdefault(_src_prefix + _mod_name[len(_top_prefix):], _mod)

del _importlib, _sys, _name, _bind_package, _ROOT_ALIASES, _mod_name, _mod, _root, _src_prefix, _top_prefix
