"""
Utility modules: dataset loaders, preprocessing helpers, etc.
"""

import sys as _sys

from .checkpointing import latest_checkpoint, maybe_load_checkpoint, safe_torch_load, save_checkpoint
from .config_io import allocate_run_dir, load_json_config, save_json_config
from .dataset_runtime import cache_path_for_entry, iter_batches, save_output_tensor, save_tensor_cache, to_2d_image
from .dataset_utils import build_dataset_from_config, build_train_val_datasets
from .distributed import (
    all_reduce_mean,
    all_reduce_tensor,
    barrier,
    broadcast_object,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)
from .evaluation_utils import latent_shape, make_grid, save_image, prepare_eval_batch
from .runtime_env import resolve_batch_size, resolve_device, set_seed, summarize_model
from . import sampling_utils
from .sampling_utils import load_run_config, resolve_checkpoint, build_sampling_dataset, resolve_output_root
from .io_utils import load, load_image, load_composite
from .dataframe_utils import lot_id
from .indexing_utils import select_visual_indices

__all__ = [
    "build_dataset_from_config",
    "build_train_val_datasets",
    "cache_path_for_entry",
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
    "get_rank",
    "get_world_size",
    "barrier",
    "broadcast_object",
    "all_reduce_tensor",
    "all_reduce_mean",
    "is_distributed",
    "is_main_process",
    "latent_shape",
    "make_grid",
    "save_image",
    "prepare_eval_batch",
    "sampling_utils",
    "load_run_config",
    "resolve_checkpoint",
    "build_sampling_dataset",
    "resolve_output_root",
    "iter_batches",
    "load",
    "load_image",
    "load_composite",
    "lot_id",
    "save_output_tensor",
    "save_tensor_cache",
    "select_visual_indices",
    "to_2d_image",
]

_prefix = f"{__name__}."
_alt_prefix = "utils." if __name__ == "src.utils" else "src.utils."
for _mod_name, _mod in list(_sys.modules.items()):
    if _mod_name.startswith(_prefix):
        _alias = _alt_prefix + _mod_name[len(_prefix):]
        _sys.modules.setdefault(_alias, _mod)
del _prefix, _alt_prefix, _mod_name, _mod, _sys
