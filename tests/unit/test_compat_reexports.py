from __future__ import annotations


def test_pipelines_utils_reexports_scheduling() -> None:
    from pipelines.utils import (
        SCHEDULER_REGISTRY,
        _prepare_attention_context,
        build_scheduler,
        normalize_latent_conditioning,
        resolve_conditioning_mode,
        resolve_scheduler_override,
        sample_with_scheduler,
        sync_if_cuda,
    )

    assert callable(build_scheduler)
    assert callable(sample_with_scheduler)
    assert callable(resolve_conditioning_mode)
    assert callable(resolve_scheduler_override)
    assert callable(sync_if_cuda)
    assert callable(normalize_latent_conditioning)
    assert callable(_prepare_attention_context)
    assert hasattr(SCHEDULER_REGISTRY, "keys")


def test_nn_losses_vae_reexports() -> None:
    from nn.losses.vae import (
        PatchDiscriminator,
        PerceptualLoss,
        bce_focal_loss,
        discriminator_hinge_loss,
        focal_loss,
        generator_hinge_loss,
    )

    assert callable(focal_loss)
    assert callable(bce_focal_loss)
    assert callable(discriminator_hinge_loss)
    assert callable(generator_hinge_loss)
    assert PatchDiscriminator is not None
    assert PerceptualLoss is not None


def test_nn_losses_vae_shim_identity() -> None:
    from nn.losses.perceptual import PerceptualLoss as direct_pl
    from nn.losses.reconstruction import focal_loss as direct_fl
    from nn.losses.vae import PerceptualLoss as shim_pl
    from nn.losses.vae import focal_loss as shim_fl

    assert shim_pl is direct_pl
    assert shim_fl is direct_fl

