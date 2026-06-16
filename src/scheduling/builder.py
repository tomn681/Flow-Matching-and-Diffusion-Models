from __future__ import annotations
from typing import Dict, Tuple

from core.noise_contracts import validate_noise_scheduler_contract
from .registry import SCHEDULER_REGISTRY, resolve_scheduler_class


_SCHEDULER_PARAM_ALLOWLIST = frozenset(
    {
        "algorithm_type",
        "beta_end",
        "beta_schedule",
        "beta_start",
        "clip_sample",
        "clip_sample_range",
        "dynamic_thresholding_ratio",
        "final_sigmas_type",
        "interpolation_type",
        "invert_sigmas",
        "lower_order_final",
        "prediction_type",
        "rescale_betas_zero_snr",
        "sample_max_value",
        "set_alpha_to_one",
        "sigma_data",
        "sigma_max",
        "sigma_min",
        "solver_order",
        "solver_type",
        "steps_offset",
        "shift",
        "thresholding",
        "timestep_spacing",
        "timestep_type",
        "trained_betas",
        "use_beta_sigmas",
        "use_exponential_sigmas",
        "use_karras_sigmas",
    }
)


def _resolve_scheduler_params(params: Dict, *, scheduler_name: str) -> Dict:
    unknown = sorted(k for k in params.keys() if k not in _SCHEDULER_PARAM_ALLOWLIST)
    if unknown:
        allowed = ", ".join(sorted(_SCHEDULER_PARAM_ALLOWLIST))
        raise ValueError(
            f"Unsupported scheduler params for '{scheduler_name}': {', '.join(unknown)}. "
            f"Allowed params: {allowed}"
        )
    return dict(params)


def resolve_conditioning_mode(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value if value else None


def build_scheduler(spec: Dict, training_cfg: Dict, *, noise_family: str | None = None) -> Tuple[object, int]:
    """Instantiate a Diffusers scheduler and return scheduler + inference steps."""
    scheduler_cfg = dict(spec or {})
    training_cfg = dict(training_cfg or {})
    name = scheduler_cfg.get("name") or training_cfg.get("scheduler") or "ddpm"
    key = str(name).lower()
    if key not in SCHEDULER_REGISTRY:
        available = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")

    cls = resolve_scheduler_class(key)
    num_train_steps = int(
        scheduler_cfg.get("num_train_timesteps")
        or training_cfg.get("num_train_timesteps")
        or 1000
    )
    top_level_params = {
        name: scheduler_cfg[name]
        for name in _SCHEDULER_PARAM_ALLOWLIST
        if name in scheduler_cfg
    }
    params = dict(top_level_params)
    params.update(dict(scheduler_cfg.get("params", {})))
    params = _resolve_scheduler_params(params, scheduler_name=key)
    family_key = str(noise_family).strip().lower() if noise_family is not None else None
    if family_key in {"diffusion", "latent_diffusion", "controlnet", "video_unet", "distillation"}:
        if key in {"euler", "euler_ancestral", "heun", "lms", "kdpm2", "kdpm2_ancestral", "dpm_multistep", "dpm_sde", "unipc"}:
            params.setdefault("use_karras_sigmas", True)
    if family_key in {"x0_denoising", "consistency"}:
        params.setdefault("prediction_type", "sample")

    scheduler = cls(num_train_timesteps=num_train_steps, **params)
    if noise_family is not None:
        validate_noise_scheduler_contract(noise_family, scheduler)
    inferred_default_steps = {
        "ddpm": 100,
        "ddim": 50,
        "pndm": 50,
        "euler": 30,
        "euler_ancestral": 30,
        "heun": 30,
        "lms": 30,
        "kdpm2": 30,
        "kdpm2_ancestral": 30,
        "deis": 20,
        "dpm_multistep": 20,
        "dpm_sde": 25,
        "unipc": 20,
        "flow_match_euler": 28,
        "flowmatch": 28,
    }
    num_inference = int(
        scheduler_cfg.get("num_inference_steps")
        or training_cfg.get("num_inference_steps")
        or inferred_default_steps.get(key, num_train_steps)
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
        "euler_karras": {"name": "euler", "params": {"use_karras_sigmas": True}},
        "euler_ancestral_karras": {"name": "euler_ancestral", "params": {"use_karras_sigmas": True}},
        "dpmpp_karras": {
            "name": "dpm_multistep",
            "params": {"solver_order": 2, "algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
        },
        "unipc_karras": {"name": "unipc", "params": {"use_karras_sigmas": True}},
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
