from __future__ import annotations

import importlib
import warnings


def test_genlib_public_api_core_exports() -> None:
    import genlib

    assert genlib.ModelFactory is not None
    assert genlib.VAETrainer is not None
    assert genlib.DiffusionTrainer is not None
    assert genlib.FlowMatchingTrainer is not None
    assert genlib.DistillationSampler is not None
    assert genlib.MODEL_REGISTRY is not None
    assert genlib.TRAINER_REGISTRY is not None
    assert genlib.SAMPLER_REGISTRY is not None
    assert genlib.SCHEDULER_REGISTRY is not None
    assert genlib.NOISE_REGISTRY is not None
    assert genlib.LOSS_REGISTRY is not None
    assert genlib.UNet2DConditionND is not None


def test_genlib_public_api_config_exports() -> None:
    import genlib

    assert callable(genlib.validate_config)
    assert callable(genlib.load_config)
    assert callable(genlib.load_and_validate)
    assert genlib.FrameworkConfig is not None
    assert genlib.TrainingConfig is not None


def test_src_is_deprecation_shim() -> None:
    import src

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(src)
    assert any("deprecated" in str(w.message).lower() for w in caught)


def test_nn_public_api_registry_exports() -> None:
    import genlib.nn as nn_pkg

    assert nn_pkg.BLOCK_REGISTRY is not None
