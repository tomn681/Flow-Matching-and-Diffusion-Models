from __future__ import annotations

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


SCHEDULER_REGISTRY: dict[str, type] = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "kdpm2_ancestral": KDPM2AncestralDiscreteScheduler,
    "deis": DEISMultistepScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "dpm_sde": DPMSolverSDEScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
    "flowmatch": FlowMatchEulerDiscreteScheduler,
}

__all__ = ["SCHEDULER_REGISTRY"]
