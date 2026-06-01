from __future__ import annotations

from sampling import SAMPLER_REGISTRY
from sampling.unet_sampler import UNetSampler


def test_unet_sampler_registry_key_present() -> None:
    assert SAMPLER_REGISTRY.get("unet") is UNetSampler

