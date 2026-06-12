"""
genlib
======

Canonical public package root for the framework.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

__version__ = "0.9.0"


def _bind_package(name: str):
    module = _importlib.import_module(f"src.{name}")
    _sys.modules[f"{__name__}.{name}"] = module
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
    "sampling",
    "scheduling",
    "training",
    "utils",
):
    _bind_package(_pkg_name)

del _pkg_name

from src.configs import FrameworkConfig, TrainingConfig, load_and_validate, load_config, validate_config
from src.core.types import ModelOutput, NoisyBatch
from src.losses import BaseLossComponent, LOSS_REGISTRY, LossAssembler
from src.models import AutoencoderKL, BaseAutoencoder, BaseVAE, ControlNetND, DiTND, MODEL_REGISTRY, ModelFactory, VQVAE, merge_models
from src.models.unet import BaseUNetND, EfficientUNetND, UNet2DConditionND, UNetDiffusersND
from src.noise import ConsistencyNoise, DDPMNoise, EDMNoise, FlowMatchingNoise, NOISE_REGISTRY, RectifiedFlowNoise, ReflowNoise, X0DenoisingNoise, generate_reflow_pairs
from src.pipelines import InferenceInputs, InferencePipeline, TextToImageInputs, TextToImagePipeline
from src.sampling import (
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
from src.scheduling import (
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
from src.training import (
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
    RectifiedFlowTrainer,
    StepwiseResolutionSchedule,
    TensorBoardCallback,
    TRAINER_REGISTRY,
    UNetTrainer,
    VAETrainer,
    VisualizationCallback,
    X0DenoisingTrainer,
    build_resolution_schedule,
)

__all__ = [
    "__version__",
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
    "ModelOutput",
    "NoisyBatch",
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
    "BaseTrainer",
    "ControlNetTrainer",
    "GenerativeTrainer",
    "DiffusionTrainer",
    "DistillationTrainer",
    "FlowMatchingTrainer",
    "ConsistencyTrainer",
    "X0DenoisingTrainer",
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
    "TensorBoardCallback",
    "VisualizationCallback",
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
    "FrameworkConfig",
    "TrainingConfig",
    "load_config",
    "load_and_validate",
    "validate_config",
    "MODEL_REGISTRY",
    "NOISE_REGISTRY",
    "LOSS_REGISTRY",
    "SCHEDULER_REGISTRY",
    "CONDITIONING_ADAPTER_REGISTRY",
    "LR_SCHEDULER_REGISTRY",
    "TRAINER_REGISTRY",
    "SAMPLER_REGISTRY",
    "BaseLossComponent",
    "LossAssembler",
]
