from __future__ import annotations

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)


SCHEDULER_REGISTRY: dict[str, type] = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "dpm_sde": DPMSolverSDEScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
    "flowmatch": FlowMatchEulerDiscreteScheduler,
}

__all__ = ["SCHEDULER_REGISTRY"]
