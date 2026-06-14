import sys as _sys

if __name__ == "src.sampling" and "sampling" in _sys.modules:
    _canonical = _sys.modules["sampling"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .base import BaseSampler
    from .controlnet_sampler import ControlNetSampler
    from .generative_sampler import (
        ConsistencySampler,
        DiffusionSampler,
        DistillationSampler,
        EDMSampler,
        FlowMatchingSampler,
        GenerativeSampler,
        ReflowSampler,
        RectifiedFlowSampler,
        VideoUNetSampler,
        X0DenoisingSampler,
    )
    from .latent_sampler import LatentDiffusionSampler, LatentFlowMatchingSampler, LatentRectifiedFlowSampler, LatentSampler
    from .registry import SAMPLER_REGISTRY
    from .unet_sampler import UNetSampler
    from .vae_sampler import VAESampler

    __all__ = [
        "BaseSampler",
        "ControlNetSampler",
        "DiffusionSampler",
        "DistillationSampler",
        "FlowMatchingSampler",
        "ConsistencySampler",
        "X0DenoisingSampler",
        "EDMSampler",
        "RectifiedFlowSampler",
        "ReflowSampler",
        "VideoUNetSampler",
        "GenerativeSampler",
        "LatentDiffusionSampler",
        "LatentFlowMatchingSampler",
        "LatentRectifiedFlowSampler",
        "LatentSampler",
        "UNetSampler",
        "SAMPLER_REGISTRY",
        "VAESampler",
    ]

    _prefix = f"{__name__}."
    _alt_prefix = "sampling." if __name__ == "src.sampling" else "src.sampling."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
