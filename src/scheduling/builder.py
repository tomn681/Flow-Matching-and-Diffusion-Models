from __future__ import annotations

import inspect
from typing import Dict, Tuple

from .registry import SCHEDULER_REGISTRY


def resolve_conditioning_mode(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value if value else None


def build_scheduler(spec: Dict, training_cfg: Dict) -> Tuple[object, int]:
    """Instantiate a Diffusers scheduler and return scheduler + inference steps."""
    scheduler_cfg = dict(spec or {})
    training_cfg = dict(training_cfg or {})
    name = scheduler_cfg.get("name") or training_cfg.get("scheduler") or "ddpm"
    key = str(name).lower()
    if key not in SCHEDULER_REGISTRY:
        available = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")

    cls = SCHEDULER_REGISTRY[key]
    num_train_steps = int(
        scheduler_cfg.get("num_train_timesteps")
        or training_cfg.get("num_train_timesteps")
        or 1000
    )
    params = dict(scheduler_cfg.get("params", {}))

    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered_params = {k: v for k, v in params.items() if k in allowed}

    scheduler = cls(num_train_timesteps=num_train_steps, **filtered_params)
    num_inference = int(
        scheduler_cfg.get("num_inference_steps")
        or training_cfg.get("num_inference_steps")
        or num_train_steps
    )
    return scheduler, num_inference


def resolve_scheduler_override(name: str | None) -> Dict | None:
    """Map user-facing scheduler aliases into scheduler config overrides."""
    if not name:
        return None
    key = str(name).strip().lower()
    if not key:
        return None

    alias = {
        "ddpm": {"name": "ddpm"},
        "ddim": {"name": "ddim"},
        "dpmsolver1": {
            "name": "dpm_multistep",
            "params": {"solver_order": 1, "algorithm_type": "dpmsolver"},
        },
        "dpmsolver2": {
            "name": "dpm_multistep",
            "params": {"solver_order": 2, "algorithm_type": "dpmsolver"},
        },
        "dpmsolver++": {
            "name": "dpm_multistep",
            "params": {"solver_order": 2, "algorithm_type": "dpmsolver++"},
        },
        "dpmsolversde": {"name": "dpm_sde"},
        "unipc": {"name": "unipc"},
        "flowmatch": {"name": "flow_match_euler"},
        "flow_match_euler": {"name": "flow_match_euler"},
    }
    if key in alias:
        return alias[key]
    if key in SCHEDULER_REGISTRY:
        return {"name": key}

    available = ", ".join(sorted(list(alias.keys())))
    raise ValueError(f"Unknown scheduler override '{name}'. Available: {available}")


__all__ = [
    "resolve_conditioning_mode",
    "build_scheduler",
    "resolve_scheduler_override",
]
