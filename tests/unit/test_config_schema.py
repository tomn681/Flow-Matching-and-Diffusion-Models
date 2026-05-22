from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from configs import load_and_validate, validate_config


def test_training_alias_warns_and_normalizes() -> None:
    raw = {
        "training": {
            "num_epochs": 3,
            "train_batch_size": 8,
            "learning_rate": 1e-4,
        },
        "model": {"model_type": "vae"},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = validate_config(raw)

    assert cfg.training.epochs == 3
    assert cfg.training.batch_size == 8
    assert any("deprecated" in str(w.message).lower() for w in caught)


def test_unknown_fields_passthrough_to_extra() -> None:
    raw = {
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "my_experiment_flag": True,
        },
        "model": {
            "model_type": "vae",
            "my_custom_model_flag": "x",
        },
        "my_top_level": 123,
    }

    cfg = validate_config(raw)
    assert cfg.training.extra["my_experiment_flag"] is True
    assert cfg.model.extra["my_custom_model_flag"] == "x"
    assert cfg.extra["my_top_level"] == 123


def test_invalid_known_value_fails() -> None:
    with pytest.raises(ValueError, match="epochs must be > 0"):
        validate_config({"training": {"epochs": 0}})


def test_dataset_only_config_is_valid() -> None:
    cfg = validate_config({"dataset_class": "datasets.base:BaseDataset"})
    assert cfg.dataset_class == "datasets.base:BaseDataset"
    assert cfg.training.epochs == 1


def test_all_json_configs_parse() -> None:
    root = Path("configs")
    json_paths = sorted(root.rglob("*.json"))
    assert json_paths

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for path in json_paths:
            cfg = load_and_validate(path)
            assert cfg is not None
