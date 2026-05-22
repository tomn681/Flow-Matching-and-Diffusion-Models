from __future__ import annotations

from configs.migration import normalize_aliases


def test_normalize_aliases_renames_known_training_keys() -> None:
    data = {"training": {"num_epochs": 10, "train_batch_size": 8}}
    result = normalize_aliases(data)
    assert result["training"]["epochs"] == 10
    assert result["training"]["batch_size"] == 8


def test_normalize_aliases_preserves_non_aliased_keys() -> None:
    data = {"training": {"keep_me": 1}}
    result = normalize_aliases(data)
    assert result["training"]["keep_me"] == 1


def test_normalize_aliases_does_not_mutate_input() -> None:
    data = {"training": {"num_epochs": 1}}
    original_data = {"training": {"num_epochs": 1}}
    normalize_aliases(data)
    assert data == original_data

