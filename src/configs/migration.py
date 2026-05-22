from __future__ import annotations

from copy import deepcopy


TRAINING_KEY_ALIASES = {
    "num_epochs": "epochs",
    "train_batch_size": "batch_size",
}


def normalize_aliases(config: dict) -> dict:
    """Return a copy of config with known aliases normalized."""
    result = deepcopy(config)
    training = result.get("training")
    if isinstance(training, dict):
        for old_key, new_key in TRAINING_KEY_ALIASES.items():
            if old_key in training and new_key not in training:
                training[new_key] = training[old_key]
    return result
