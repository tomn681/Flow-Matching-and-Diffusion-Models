from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelFamily:
    key: str
    model_types: tuple[str, ...]
    trainer_key: str | None
    sampler_key: str | None
    noise_family: str | None
    runtime_kind: str
    latent_capable: bool = False
    prediction_types: tuple[str, ...] = ("epsilon",)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    plugin: str = "builtin"


class ModelFamilyRegistry:
    def __init__(self) -> None:
        self._by_key: dict[str, ModelFamily] = {}
        self._model_type_to_key: dict[str, str] = {}

    def register(self, family: ModelFamily) -> ModelFamily:
        key = str(family.key).strip().lower()
        if not key:
            raise ValueError("Model family key cannot be empty.")
        if key in self._by_key:
            raise ValueError(f"Model family '{key}' is already registered.")
        normalized_types = tuple(str(t).strip().lower() for t in family.model_types)
        for model_type in normalized_types:
            if not model_type:
                raise ValueError("Model family model_types cannot contain empty entries.")
            if model_type in self._model_type_to_key:
                raise ValueError(
                    f"Model type '{model_type}' is already claimed by family "
                    f"'{self._model_type_to_key[model_type]}'."
                )
        normalized_family = ModelFamily(
            key=key,
            model_types=normalized_types,
            trainer_key=family.trainer_key,
            sampler_key=family.sampler_key,
            noise_family=family.noise_family,
            runtime_kind=str(family.runtime_kind).strip().lower(),
            latent_capable=bool(family.latent_capable),
            prediction_types=tuple(str(v).strip().lower() for v in family.prediction_types),
            capabilities=frozenset(str(v).strip().lower() for v in family.capabilities),
            plugin=family.plugin,
        )
        self._by_key[key] = normalized_family
        for model_type in normalized_types:
            self._model_type_to_key[model_type] = key
        return normalized_family

    def get(self, key: str) -> ModelFamily:
        lookup = str(key).strip().lower()
        if lookup not in self._by_key:
            available = ", ".join(sorted(self._by_key.keys()))
            raise KeyError(f"Unknown model family '{key}'. Available: [{available}]")
        return self._by_key[lookup]

    def for_model_type(self, model_type: str | None) -> ModelFamily | None:
        if model_type is None:
            return None
        lookup = str(model_type).strip().lower()
        family_key = self._model_type_to_key.get(lookup)
        if family_key is None:
            return None
        return self._by_key[family_key]

    def list(self) -> list[str]:
        return sorted(self._by_key.keys())

    def items(self) -> list[tuple[str, ModelFamily]]:
        return [(key, self._by_key[key]) for key in self.list()]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.strip().lower() in self._by_key

    def __len__(self) -> int:
        return len(self._by_key)


MODEL_FAMILY_REGISTRY = ModelFamilyRegistry()
_FAMILIES_LOADED = False


def ensure_model_families_loaded() -> None:
    global _FAMILIES_LOADED
    if _FAMILIES_LOADED:
        return
    from core.plugin import RegistryHub, load_plugins
    from importlib import import_module

    hub = RegistryHub(model_families=MODEL_FAMILY_REGISTRY)
    try:
        register_builtin = import_module("plugins.builtin").register
    except Exception:
        register_builtin = import_module("src.plugins.builtin").register
    register_builtin(hub)
    load_plugins(hub=hub)
    _FAMILIES_LOADED = True


def get_model_family(key: str) -> ModelFamily:
    ensure_model_families_loaded()
    return MODEL_FAMILY_REGISTRY.get(key)


def model_family_for_model_type(model_type: str | None) -> ModelFamily | None:
    ensure_model_families_loaded()
    return MODEL_FAMILY_REGISTRY.for_model_type(model_type)


__all__ = [
    "ModelFamily",
    "ModelFamilyRegistry",
    "MODEL_FAMILY_REGISTRY",
    "ensure_model_families_loaded",
    "get_model_family",
    "model_family_for_model_type",
]
