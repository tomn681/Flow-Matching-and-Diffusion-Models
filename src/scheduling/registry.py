from __future__ import annotations

from core.registry import Registry

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


SCHEDULER_REGISTRY = Registry[type]("schedulers")

for _name, _cls in (
    ("ddpm", DDPMScheduler),
    ("ddim", DDIMScheduler),
    ("pndm", PNDMScheduler),
    ("euler", EulerDiscreteScheduler),
    ("euler_ancestral", EulerAncestralDiscreteScheduler),
    ("heun", HeunDiscreteScheduler),
    ("lms", LMSDiscreteScheduler),
    ("kdpm2", KDPM2DiscreteScheduler),
    ("kdpm2_ancestral", KDPM2AncestralDiscreteScheduler),
    ("deis", DEISMultistepScheduler),
    ("dpm_multistep", DPMSolverMultistepScheduler),
    ("dpm_sde", DPMSolverSDEScheduler),
    ("unipc", UniPCMultistepScheduler),
    ("flow_match_euler", FlowMatchEulerDiscreteScheduler),
    ("flowmatch", FlowMatchEulerDiscreteScheduler),
):
    SCHEDULER_REGISTRY.register_value(_name, _cls)

del _name, _cls

__all__ = ["SCHEDULER_REGISTRY"]
