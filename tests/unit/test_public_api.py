from __future__ import annotations

import importlib
import warnings
from pathlib import Path


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


def test_genlib_import_does_not_inject_src_root_into_sys_path() -> None:
    root = Path(__file__).resolve().parents[2]
    run_model_wrapper = (root / "genlib" / "run_model.py").read_text(encoding="utf-8")
    train_wrapper = (root / "genlib" / "train.py").read_text(encoding="utf-8")

    assert "from src import run_model" not in run_model_wrapper
    assert "from src import train" not in train_wrapper


def test_flat_root_aliases_resolve_to_genlib_modules() -> None:
    import genlib.models as gen_models
    import models

    assert models is gen_models
