"""
Generative Modeling Framework
=============================

Curated public API across model definitions, training orchestration,
noise/scheduler abstractions, and compatibility wrappers.
"""

from . import compat, configs, core, datasets, losses, models, nn, noise, pipelines, scheduling, training, utils
from .configs import FrameworkConfig, TrainingConfig, load_and_validate, load_config, validate_config
from .core.types import ModelOutput, NoisyBatch
from .losses import BaseLossComponent, LOSS_REGISTRY, LossAssembler
from .models import AutoencoderKL, BaseAutoencoder, BaseVAE, ControlNetND, MODEL_REGISTRY, ModelFactory, VQVAE
from .models.unet import BaseUNetND, EfficientUNetND, UNet2DConditionND, UNetDiffusersND
from .noise import DDPMNoise, FlowMatchingNoise, NOISE_REGISTRY
from .pipelines import InferenceInputs, InferencePipeline
from .sampling import BaseSampler, DiffusionSampler, FlowMatchingSampler, SAMPLER_REGISTRY, VAESampler
from .scheduling import (
    CONDITIONING_ADAPTER_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    SCHEDULER_REGISTRY,
    build_lr_scheduler,
    build_scheduler,
    resolve_conditioning_adapter,
    resolve_conditioning_mode,
    resolve_scheduler_override,
    sample_with_scheduler,
)
from .training import (
    BaseTrainer,
    CheckpointCallback,
    DiffusionTrainer,
    FlowMatchingTrainer,
    GenerativeTrainer,
    MetricsCSVCallback,
    TRAINER_REGISTRY,
    VAETrainer,
    VisualizationCallback,
)

__all__ = [
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
    "ControlNetND",
    "InferenceInputs",
    "InferencePipeline",
    # Training
    "BaseTrainer",
    "GenerativeTrainer",
    "DiffusionTrainer",
    "FlowMatchingTrainer",
    "VAETrainer",
    "CheckpointCallback",
    "MetricsCSVCallback",
    "VisualizationCallback",
    # Noise / scheduling / sampling
    "DDPMNoise",
    "FlowMatchingNoise",
    "build_scheduler",
    "build_lr_scheduler",
    "resolve_conditioning_mode",
    "resolve_conditioning_adapter",
    "resolve_scheduler_override",
    "sample_with_scheduler",
    "BaseSampler",
    "VAESampler",
    "DiffusionSampler",
    "FlowMatchingSampler",
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
for _name in ("nn", "pipelines", "models", "utils", "core", "noise", "losses", "configs", "scheduling", "training", "datasets", "compat"):
    _sys.modules.setdefault(_name, _sys.modules[f"{__package__}.{_name}"])
del _sys, _name
