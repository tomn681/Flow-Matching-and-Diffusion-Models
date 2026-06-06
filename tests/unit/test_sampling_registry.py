from __future__ import annotations

from sampling import (
    ConsistencySampler,
    ControlNetSampler,
    DiffusionSampler,
    DistillationSampler,
    EDMSampler,
    FlowMatchingSampler,
    LatentDiffusionSampler,
    LatentFlowMatchingSampler,
    LatentRectifiedFlowSampler,
    ReflowSampler,
    RectifiedFlowSampler,
    SAMPLER_REGISTRY,
    VAESampler,
    VideoUNetSampler,
)


def test_sampler_registry_entries() -> None:
    keys = set(SAMPLER_REGISTRY.list())
    assert {
        "vae",
        "diffusion",
        "distillation",
        "flow_matching",
        "consistency",
        "controlnet",
        "edm",
        "rectified_flow",
        "reflow",
        "video_unet",
        "latent_diffusion",
        "latent_flow_matching",
        "latent_rectified_flow",
    }.issubset(keys)


def test_sampler_registry_maps_to_expected_classes() -> None:
    assert SAMPLER_REGISTRY.get("vae") is VAESampler
    assert SAMPLER_REGISTRY.get("diffusion") is DiffusionSampler
    assert SAMPLER_REGISTRY.get("distillation") is DistillationSampler
    assert SAMPLER_REGISTRY.get("flow_matching") is FlowMatchingSampler
    assert SAMPLER_REGISTRY.get("consistency") is ConsistencySampler
    assert SAMPLER_REGISTRY.get("controlnet") is ControlNetSampler
    assert SAMPLER_REGISTRY.get("edm") is EDMSampler
    assert SAMPLER_REGISTRY.get("rectified_flow") is RectifiedFlowSampler
    assert SAMPLER_REGISTRY.get("reflow") is ReflowSampler
    assert SAMPLER_REGISTRY.get("video_unet") is VideoUNetSampler
    assert SAMPLER_REGISTRY.get("latent_diffusion") is LatentDiffusionSampler
    assert SAMPLER_REGISTRY.get("latent_flow_matching") is LatentFlowMatchingSampler
    assert SAMPLER_REGISTRY.get("latent_rectified_flow") is LatentRectifiedFlowSampler
