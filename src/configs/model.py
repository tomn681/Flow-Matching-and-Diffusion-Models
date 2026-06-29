from __future__ import annotations

from dataclasses import dataclass, fields

from .base import BaseConfig


@dataclass
class BaseModelConfig(BaseConfig):
    model_type: str = ""


@dataclass
class VAEModelConfig(BaseModelConfig):
    latent_type: str = "kl"
    recon_type: str = "l1"


@dataclass
class DiffusionModelConfig(BaseModelConfig):
    conditioning: str | None = None


@dataclass
class FlowMatchingModelConfig(BaseModelConfig):
    conditioning: str | None = None


_MODEL_CLASS_BY_TYPE = {
    "vae": VAEModelConfig,
    "diffusion": DiffusionModelConfig,
    "flow_matching": FlowMatchingModelConfig,
    "rectified_flow": FlowMatchingModelConfig,
    "residual_flow_matching": FlowMatchingModelConfig,
    "residual_rectified_flow": FlowMatchingModelConfig,
    "latent_flow_matching": FlowMatchingModelConfig,
    "latent_rectified_flow": FlowMatchingModelConfig,
}


def _from_dict_typed(data: dict, cls: type[BaseModelConfig]) -> BaseModelConfig:
    known_fields = {f.name for f in fields(cls)}
    normalized: dict = {}
    extra: dict = {}

    for key, value in data.items():
        if key in known_fields:
            normalized[key] = value
        else:
            extra[key] = value

    normalized["extra"] = extra
    return cls(**normalized)


def build_model_config(data: dict | None) -> BaseModelConfig:
    if data is None:
        return BaseModelConfig()
    if not isinstance(data, dict):
        raise TypeError(f"model must be a dict, got {type(data).__name__}")

    model_type = str(data.get("model_type", "")).lower()
    model_cls = _MODEL_CLASS_BY_TYPE.get(model_type, BaseModelConfig)
    return _from_dict_typed(data, model_cls)
