"""
Utility modules: dataset loaders, preprocessing helpers, etc.
"""

from .dataset_runtime import cache_path_for_entry, iter_batches, save_output_tensor, save_tensor_cache, to_2d_image
from .dataset_utils import build_dataset_from_config, build_train_val_datasets
from .training_utils import (
    load_json_config,
    save_json_config,
    safe_torch_load,
    set_seed,
    resolve_device,
    resolve_batch_size,
    summarize_model,
    allocate_run_dir,
    latest_checkpoint,
    save_checkpoint,
    maybe_load_checkpoint,
    setup_distributed,
    is_distributed,
    is_main_process,
)
from .evaluation_utils import latent_shape, make_grid, save_image, prepare_eval_batch
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
    "is_distributed",
    "is_main_process",
    "latent_shape",
    "make_grid",
    "save_image",
    "prepare_eval_batch",
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
