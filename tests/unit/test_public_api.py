from __future__ import annotations


def test_src_public_api_core_exports() -> None:
    import src

    assert src.ModelFactory is not None
    assert src.VAETrainer is not None
    assert src.DiffusionTrainer is not None
    assert src.FlowMatchingTrainer is not None
    assert src.MODEL_REGISTRY is not None
    assert src.TRAINER_REGISTRY is not None
    assert src.SAMPLER_REGISTRY is not None
    assert src.SCHEDULER_REGISTRY is not None
    assert src.NOISE_REGISTRY is not None
    assert src.LOSS_REGISTRY is not None


def test_src_public_api_config_exports() -> None:
    import src

    assert callable(src.validate_config)
    assert callable(src.load_config)
    assert callable(src.load_and_validate)
    assert src.FrameworkConfig is not None
    assert src.TrainingConfig is not None

