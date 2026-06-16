"""Built-in plugin entrypoint target used by `genlib.plugins` discovery."""

from __future__ import annotations


def _builtin_families():
    from core.families import ModelFamily

    return [
        ModelFamily("vae", ("vae",), "vae", "vae", None, runtime_kind="autoencoder", prediction_types=("sample",)),
        ModelFamily("diffusion", ("diffusion", "video_unet", "distillation"), "diffusion", "diffusion", "ddpm", runtime_kind="generative", prediction_types=("epsilon", "sample", "v_prediction")),
        ModelFamily("flow_matching", ("flow_matching",), "flow_matching", "flow_matching", "flow_matching", runtime_kind="generative", prediction_types=("epsilon",)),
        ModelFamily("rectified_flow", ("rectified_flow",), "rectified_flow", "rectified_flow", "rectified_flow", runtime_kind="generative", prediction_types=("epsilon",)),
        ModelFamily("reflow", ("reflow",), "reflow", "reflow", "reflow", runtime_kind="generative", prediction_types=("epsilon",)),
        ModelFamily("x0_denoising", ("x0_denoising", "consistency"), "x0_denoising", "x0_denoising", "x0_denoising", runtime_kind="generative", prediction_types=("sample",)),
        ModelFamily("edm", ("edm",), "edm", "edm", "edm", runtime_kind="generative", prediction_types=("epsilon",)),
        ModelFamily("controlnet", ("controlnet",), "controlnet", "controlnet", "ddpm", runtime_kind="controlnet", prediction_types=("epsilon", "sample", "v_prediction")),
        ModelFamily("latent_diffusion", ("latent_diffusion",), "latent_diffusion", "latent_diffusion", "ddpm", runtime_kind="latent", latent_capable=True, prediction_types=("epsilon", "sample", "v_prediction")),
        ModelFamily("latent_flow_matching", ("latent_flow_matching",), "latent_flow_matching", "latent_flow_matching", "flow_matching", runtime_kind="latent", latent_capable=True, prediction_types=("epsilon",)),
        ModelFamily("latent_rectified_flow", ("latent_rectified_flow",), "latent_rectified_flow", "latent_rectified_flow", "rectified_flow", runtime_kind="latent", latent_capable=True, prediction_types=("epsilon",)),
        ModelFamily("gan", ("gan",), "gan", None, None, runtime_kind="trainer_only", prediction_types=("sample",)),
        ModelFamily("unet", ("unet",), "unet", "unet", "ddpm", runtime_kind="generative", prediction_types=("epsilon", "sample", "v_prediction")),
        ModelFamily("patch_transformer", ("patch_transformer", "dit"), None, None, None, runtime_kind="model_only", prediction_types=("epsilon",)),
    ]


def register(hub=None) -> str:
    """Register built-in model-family descriptors when a registry hub is provided."""
    if hub is not None and getattr(hub, "model_families", None) is not None:
        registry = hub.model_families
        for family in _builtin_families():
            if family.key not in registry:
                registry.register(family)
    return "builtin"
