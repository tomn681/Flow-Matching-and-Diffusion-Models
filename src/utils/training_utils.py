from __future__ import annotations

"""
Compatibility re-export layer for training/runtime utility helpers.

Canonical ownership now lives in:
- `utils.config_io`
- `utils.runtime_env`
- `utils.checkpointing`
- `utils.distributed`
"""

from .checkpointing import latest_checkpoint, maybe_load_checkpoint, safe_torch_load, save_checkpoint
from .config_io import allocate_run_dir, load_json_config, save_json_config
from .distributed import is_distributed, is_main_process, setup_distributed
from .runtime_env import resolve_batch_size, resolve_device, set_seed, summarize_model

__all__ = [
    "load_json_config",
    "save_json_config",
    "safe_torch_load",
    "set_seed",
    "resolve_device",
    "resolve_batch_size",
    "summarize_model",
    "allocate_run_dir",
    "latest_checkpoint",
    "save_checkpoint",
    "maybe_load_checkpoint",
    "setup_distributed",
    "is_distributed",
    "is_main_process",
]
