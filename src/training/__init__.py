import sys as _sys

if __name__ == "src.training" and "training" in _sys.modules:
    _canonical = _sys.modules["training"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .base import BaseTrainer
    from .builder import TrainerBuilder
    from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback, VisualizationCallback
    from .controlnet_trainer import ControlNetTrainer
    from .distillation_trainer import DistillationTrainer
    from .ema import EMAModel
    from .events import TrainingEventBus
    from .gan_trainer import GANTrainer
    from .generative_trainer import (
        ConsistencyTrainer,
        DiffusionTrainer,
        EDMTrainer,
        FlowMatchingTrainer,
        GenerativeTrainer,
        ReflowTrainer,
        RectifiedFlowTrainer,
        X0DenoisingTrainer,
    )
    from .latent_trainer import LatentDiffusionTrainer, LatentFlowMatchingTrainer, LatentGenerativeTrainer, LatentRectifiedFlowTrainer
    from .lora import LoRAWrapper, load_lora_weights, save_lora_weights, wrap_lora
    from .multi_resolution import StepwiseResolutionSchedule, build_resolution_schedule
    from .registry import TRAINER_REGISTRY
    from .unet_trainer import UNetTrainer
    from .vae_trainer import VAETrainer

    TRAINER_REGISTRY.set_base_type(BaseTrainer)

    __all__ = [
        "BaseTrainer",
        "CheckpointCallback",
        "ControlNetTrainer",
        "DiffusionTrainer",
        "DistillationTrainer",
        "FlowMatchingTrainer",
        "ConsistencyTrainer",
        "X0DenoisingTrainer",
        "EDMTrainer",
        "RectifiedFlowTrainer",
        "ReflowTrainer",
        "GANTrainer",
        "GenerativeTrainer",
        "LatentGenerativeTrainer",
        "LatentDiffusionTrainer",
        "LatentFlowMatchingTrainer",
        "LatentRectifiedFlowTrainer",
        "LoRAWrapper",
        "wrap_lora",
        "save_lora_weights",
        "load_lora_weights",
        "MetricsCSVCallback",
        "TensorBoardCallback",
        "EMAModel",
        "TrainerBuilder",
        "TrainingEventBus",
        "TRAINER_REGISTRY",
        "StepwiseResolutionSchedule",
        "build_resolution_schedule",
        "UNetTrainer",
        "VAETrainer",
        "VisualizationCallback",
    ]

    _prefix = f"{__name__}."
    _alt_prefix = "training." if __name__ == "src.training" else "src.training."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
