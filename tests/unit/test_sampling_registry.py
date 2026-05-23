from __future__ import annotations

from sampling import (
    DiffusionSampler,
    FlowMatchingSampler,
    LatentDiffusionSampler,
    LatentFlowMatchingSampler,
    SAMPLER_REGISTRY,
    VAESampler,
)


def test_sampler_registry_entries() -> None:
    keys = set(SAMPLER_REGISTRY.list())
    assert {"vae", "diffusion", "flow_matching", "latent_diffusion", "latent_flow_matching"}.issubset(keys)


def test_sampler_registry_maps_to_expected_classes() -> None:
    assert SAMPLER_REGISTRY.get("vae") is VAESampler
    assert SAMPLER_REGISTRY.get("diffusion") is DiffusionSampler
    assert SAMPLER_REGISTRY.get("flow_matching") is FlowMatchingSampler
    assert SAMPLER_REGISTRY.get("latent_diffusion") is LatentDiffusionSampler
    assert SAMPLER_REGISTRY.get("latent_flow_matching") is LatentFlowMatchingSampler
