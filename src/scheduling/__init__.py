import sys as _sys

if __name__ == "src.scheduling" and "scheduling" in _sys.modules:
    _canonical = _sys.modules["scheduling"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .builder import build_scheduler, resolve_conditioning_mode, resolve_scheduler_override
    from .conditioning import (
        CONDITIONING_ADAPTER_REGISTRY,
        LatentAttentionAdapter,
        resolve_conditioning_adapter,
    )
    from .conditioning_chain import ChainAdapterSpec, ConditioningChain
    from .text_conditioning import TextConditioningAdapter, build_text_conditioning_adapter
    from .lr import LR_SCHEDULER_REGISTRY, build_lr_scheduler
    from .registry import SCHEDULER_REGISTRY
    from .sampling_loop import (
        _prepare_attention_context,
        normalize_latent_conditioning,
        sample_with_scheduler,
        sync_if_cuda,
    )

    __all__ = [
        "SCHEDULER_REGISTRY",
        "CONDITIONING_ADAPTER_REGISTRY",
        "LR_SCHEDULER_REGISTRY",
        "build_scheduler",
        "build_lr_scheduler",
        "LatentAttentionAdapter",
        "ChainAdapterSpec",
        "ConditioningChain",
        "TextConditioningAdapter",
        "build_text_conditioning_adapter",
        "resolve_conditioning_adapter",
        "resolve_conditioning_mode",
        "resolve_scheduler_override",
        "normalize_latent_conditioning",
        "_prepare_attention_context",
        "sample_with_scheduler",
        "sync_if_cuda",
    ]

    _prefix = f"{__name__}."
    _alt_prefix = "scheduling." if __name__ == "src.scheduling" else "src.scheduling."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
