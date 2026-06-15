"""
genlib
======

Canonical public package root for the framework.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path

__version__ = "0.9.0"

_SRC_ROOT = _Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in __path__:
    __path__.append(str(_SRC_ROOT))
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

_PACKAGE_NAMES = {
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
}

_EXPORT_MAP = {
    "FrameworkConfig": ("genlib.configs", "FrameworkConfig"),
    "TrainingConfig": ("genlib.configs", "TrainingConfig"),
    "load_and_validate": ("genlib.configs", "load_and_validate"),
    "load_config": ("genlib.configs", "load_config"),
    "validate_config": ("genlib.configs", "validate_config"),
    "ModelOutput": ("genlib.core.types", "ModelOutput"),
    "NoisyBatch": ("genlib.core.types", "NoisyBatch"),
    "BaseLossComponent": ("genlib.losses", "BaseLossComponent"),
    "LOSS_REGISTRY": ("genlib.losses", "LOSS_REGISTRY"),
    "LossAssembler": ("genlib.losses", "LossAssembler"),
    "AutoencoderKL": ("genlib.models", "AutoencoderKL"),
    "BaseAutoencoder": ("genlib.models", "BaseAutoencoder"),
    "BaseVAE": ("genlib.models", "BaseVAE"),
    "ControlNetND": ("genlib.models", "ControlNetND"),
    "DiTND": ("genlib.models", "DiTND"),
    "MODEL_REGISTRY": ("genlib.models", "MODEL_REGISTRY"),
    "ModelFactory": ("genlib.models", "ModelFactory"),
    "VQVAE": ("genlib.models", "VQVAE"),
    "merge_models": ("genlib.models", "merge_models"),
    "BaseUNetND": ("genlib.models.unet", "BaseUNetND"),
    "EfficientUNetND": ("genlib.models.unet", "EfficientUNetND"),
    "UNet2DConditionND": ("genlib.models.unet", "UNet2DConditionND"),
    "UNetDiffusersND": ("genlib.models.unet", "UNetDiffusersND"),
    "ConsistencyNoise": ("genlib.noise", "ConsistencyNoise"),
    "DDPMNoise": ("genlib.noise", "DDPMNoise"),
    "EDMNoise": ("genlib.noise", "EDMNoise"),
    "FlowMatchingNoise": ("genlib.noise", "FlowMatchingNoise"),
    "NOISE_REGISTRY": ("genlib.noise", "NOISE_REGISTRY"),
    "RectifiedFlowNoise": ("genlib.noise", "RectifiedFlowNoise"),
    "ReflowNoise": ("genlib.noise", "ReflowNoise"),
    "X0DenoisingNoise": ("genlib.noise", "X0DenoisingNoise"),
    "generate_reflow_pairs": ("genlib.noise", "generate_reflow_pairs"),
    "InferenceInputs": ("genlib.pipelines", "InferenceInputs"),
    "InferencePipeline": ("genlib.pipelines", "InferencePipeline"),
    "TextToImageInputs": ("genlib.pipelines", "TextToImageInputs"),
    "TextToImagePipeline": ("genlib.pipelines", "TextToImagePipeline"),
    "BaseSampler": ("genlib.sampling", "BaseSampler"),
    "ConsistencySampler": ("genlib.sampling", "ConsistencySampler"),
    "ControlNetSampler": ("genlib.sampling", "ControlNetSampler"),
    "DiffusionSampler": ("genlib.sampling", "DiffusionSampler"),
    "DistillationSampler": ("genlib.sampling", "DistillationSampler"),
    "EDMSampler": ("genlib.sampling", "EDMSampler"),
    "FlowMatchingSampler": ("genlib.sampling", "FlowMatchingSampler"),
    "LatentDiffusionSampler": ("genlib.sampling", "LatentDiffusionSampler"),
    "LatentFlowMatchingSampler": ("genlib.sampling", "LatentFlowMatchingSampler"),
    "LatentRectifiedFlowSampler": ("genlib.sampling", "LatentRectifiedFlowSampler"),
    "ReflowSampler": ("genlib.sampling", "ReflowSampler"),
    "RectifiedFlowSampler": ("genlib.sampling", "RectifiedFlowSampler"),
    "SAMPLER_REGISTRY": ("genlib.sampling", "SAMPLER_REGISTRY"),
    "UNetSampler": ("genlib.sampling", "UNetSampler"),
    "VAESampler": ("genlib.sampling", "VAESampler"),
    "VideoUNetSampler": ("genlib.sampling", "VideoUNetSampler"),
    "CONDITIONING_ADAPTER_REGISTRY": ("genlib.scheduling", "CONDITIONING_ADAPTER_REGISTRY"),
    "LR_SCHEDULER_REGISTRY": ("genlib.scheduling", "LR_SCHEDULER_REGISTRY"),
    "SCHEDULER_REGISTRY": ("genlib.scheduling", "SCHEDULER_REGISTRY"),
    "TextConditioningAdapter": ("genlib.scheduling", "TextConditioningAdapter"),
    "build_lr_scheduler": ("genlib.scheduling", "build_lr_scheduler"),
    "build_scheduler": ("genlib.scheduling", "build_scheduler"),
    "build_text_conditioning_adapter": ("genlib.scheduling", "build_text_conditioning_adapter"),
    "resolve_conditioning_adapter": ("genlib.scheduling", "resolve_conditioning_adapter"),
    "resolve_conditioning_mode": ("genlib.scheduling", "resolve_conditioning_mode"),
    "resolve_scheduler_override": ("genlib.scheduling", "resolve_scheduler_override"),
    "sample_with_scheduler": ("genlib.scheduling", "sample_with_scheduler"),
    "BaseTrainer": ("genlib.training", "BaseTrainer"),
    "CheckpointCallback": ("genlib.training", "CheckpointCallback"),
    "ConsistencyTrainer": ("genlib.training", "ConsistencyTrainer"),
    "ControlNetTrainer": ("genlib.training", "ControlNetTrainer"),
    "DiffusionTrainer": ("genlib.training", "DiffusionTrainer"),
    "DistillationTrainer": ("genlib.training", "DistillationTrainer"),
    "EDMTrainer": ("genlib.training", "EDMTrainer"),
    "FlowMatchingTrainer": ("genlib.training", "FlowMatchingTrainer"),
    "GANTrainer": ("genlib.training", "GANTrainer"),
    "GenerativeTrainer": ("genlib.training", "GenerativeTrainer"),
    "LatentDiffusionTrainer": ("genlib.training", "LatentDiffusionTrainer"),
    "LatentFlowMatchingTrainer": ("genlib.training", "LatentFlowMatchingTrainer"),
    "LatentGenerativeTrainer": ("genlib.training", "LatentGenerativeTrainer"),
    "LatentRectifiedFlowTrainer": ("genlib.training", "LatentRectifiedFlowTrainer"),
    "LoRAWrapper": ("genlib.training", "LoRAWrapper"),
    "MetricsCSVCallback": ("genlib.training", "MetricsCSVCallback"),
    "ReflowTrainer": ("genlib.training", "ReflowTrainer"),
    "RectifiedFlowTrainer": ("genlib.training", "RectifiedFlowTrainer"),
    "StepwiseResolutionSchedule": ("genlib.training", "StepwiseResolutionSchedule"),
    "TensorBoardCallback": ("genlib.training", "TensorBoardCallback"),
    "TRAINER_REGISTRY": ("genlib.training", "TRAINER_REGISTRY"),
    "UNetTrainer": ("genlib.training", "UNetTrainer"),
    "VAETrainer": ("genlib.training", "VAETrainer"),
    "VisualizationCallback": ("genlib.training", "VisualizationCallback"),
    "X0DenoisingTrainer": ("genlib.training", "X0DenoisingTrainer"),
    "build_resolution_schedule": ("genlib.training", "build_resolution_schedule"),
}


def _bind_package(name: str):
    module = _importlib.import_module(name)
    _sys.modules[name] = module
    _sys.modules[f"{__name__}.{name}"] = module
    _sys.modules[f"src.{name}"] = module

    prefixes = (f"{name}.", f"src.{name}.", f"{__name__}.{name}.")
    for mod_name, mod in list(_sys.modules.items()):
        if mod_name.startswith(f"{name}."):
            suffix = mod_name[len(name) + 1 :]
            _sys.modules.setdefault(f"src.{name}.{suffix}", mod)
            _sys.modules.setdefault(f"{__name__}.{name}.{suffix}", mod)
        elif mod_name.startswith(f"src.{name}."):
            suffix = mod_name[len(f'src.{name}.') :]
            _sys.modules.setdefault(f"{name}.{suffix}", mod)
            _sys.modules.setdefault(f"{__name__}.{name}.{suffix}", mod)
        elif mod_name.startswith(f"{__name__}.{name}."):
            suffix = mod_name[len(f'{__name__}.{name}.') :]
            _sys.modules.setdefault(f"{name}.{suffix}", mod)
            _sys.modules.setdefault(f"src.{name}.{suffix}", mod)
    return module


for _pkg_name in sorted(_PACKAGE_NAMES):
    _bind_package(_pkg_name)

del _pkg_name


def __getattr__(name: str):
    if name in _PACKAGE_NAMES:
        return _sys.modules[f"{__name__}.{name}"]
    if name in _EXPORT_MAP:
        module_name, attr = _EXPORT_MAP[name]
        return getattr(_importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    *_EXPORT_MAP.keys(),
]
