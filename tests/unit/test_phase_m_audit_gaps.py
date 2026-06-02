from __future__ import annotations

from sampling import SAMPLER_REGISTRY
from training import TRAINER_REGISTRY


def test_trainer_sampler_registry_symmetry() -> None:
    trainer_keys = set(TRAINER_REGISTRY.list())
    sampler_keys = set(SAMPLER_REGISTRY.list())
    excluded = {"gan"}
    assert trainer_keys - excluded <= sampler_keys
