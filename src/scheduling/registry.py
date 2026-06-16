from __future__ import annotations

from importlib import import_module

from core.registry import Registry


SCHEDULER_REGISTRY = Registry[str]("schedulers")

_SCHEDULER_IMPORTS = {
    "ddpm": "diffusers:DDPMScheduler",
    "ddim": "diffusers:DDIMScheduler",
    "pndm": "diffusers:PNDMScheduler",
    "euler": "diffusers:EulerDiscreteScheduler",
    "euler_ancestral": "diffusers:EulerAncestralDiscreteScheduler",
    "heun": "diffusers:HeunDiscreteScheduler",
    "lms": "diffusers:LMSDiscreteScheduler",
    "kdpm2": "diffusers:KDPM2DiscreteScheduler",
    "kdpm2_ancestral": "diffusers:KDPM2AncestralDiscreteScheduler",
    "deis": "diffusers:DEISMultistepScheduler",
    "dpm_multistep": "diffusers:DPMSolverMultistepScheduler",
    "dpm_sde": "diffusers:DPMSolverSDEScheduler",
    "unipc": "diffusers:UniPCMultistepScheduler",
    "flow_match_euler": "diffusers:FlowMatchEulerDiscreteScheduler",
    "flowmatch": "diffusers:FlowMatchEulerDiscreteScheduler",
}

for _name, _target in _SCHEDULER_IMPORTS.items():
    SCHEDULER_REGISTRY.register_value(_name, _target)


def resolve_scheduler_class(name: str):
    key = str(name).strip().lower()
    target = SCHEDULER_REGISTRY.get(key)
    if not isinstance(target, str):
        return target
    module_name, attr = target.split(":", 1)
    module = import_module(module_name)
    cls = getattr(module, attr)
    SCHEDULER_REGISTRY._entries[key] = cls
    return cls


__all__ = ["SCHEDULER_REGISTRY", "resolve_scheduler_class"]
