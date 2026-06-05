"""
Generative Modeling Framework
=============================

Curated public API across model definitions, training orchestration,
noise/scheduler abstractions, and compatibility wrappers.
"""

__version__ = "1.0.0"

from . import compat, configs, core, datasets, losses, models, nn, noise, pipelines, plugins, scheduling, training, utils
from .configs import FrameworkConfig, TrainingConfig, load_and_validate, load_config, validate_config
from .core.types import ModelOutput, NoisyBatch
from .losses import BaseLossComponent, LOSS_REGISTRY, LossAssembler
from .models import AutoencoderKL, BaseAutoencoder, BaseVAE, ControlNetND, DiTND, MODEL_REGISTRY, ModelFactory, VQVAE, merge_models
from .models.unet import BaseUNetND, EfficientUNetND, UNet2DConditionND, UNetDiffusersND
from .noise import ConsistencyNoise, DDPMNoise, EDMNoise, FlowMatchingNoise, NOISE_REGISTRY, RectifiedFlowNoise, ReflowNoise, generate_reflow_pairs
from .pipelines import InferenceInputs, InferencePipeline, TextToImageInputs, TextToImagePipeline
from .sampling import (
    BaseSampler,
    ConsistencySampler,
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
import sys as _sys
for _name in ("nn", "pipelines", "models", "utils", "core", "noise", "losses", "configs", "scheduling", "training", "datasets", "compat", "plugins"):
    _sys.modules.setdefault(_name, _sys.modules[f"{__package__}.{_name}"])
del _sys, _name
