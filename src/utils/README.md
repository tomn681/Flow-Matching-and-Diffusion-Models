# `src.utils`

Utilities supporting data loading and experiment management.

## Datasets

Dataset implementations live under `src/datasets/`:
- `BaseDataset`, `LDCTDataset`, and `MNISTDataset` with full docstrings in that package.

### `build_dataset_from_config` / `build_train_val_datasets`
Config-driven dataset builders used by the training pipeline. Dataset classes are resolved from each config's `dataset.class` (or legacy `dataset.dataset_class`) field.

## Utility split

- `io_utils.py`: `load`, `load_image`, `load_composite` (DICOM/NumPy/Tensor/image IO).
- `dataframe_utils.py`: `lot_id` helpers for deterministic case/slice IDs.
- `indexing_utils.py`: sample-selection helpers like `select_visual_indices`.
- `dataset_runtime.py`: cache-path generation, simple batch iteration, and
  output-tensor persistence for dataset-backed runtime/export flows.
- `utils.py`: compatibility shim re-exporting the functions above.

## Utility Split

The old `training_utils.py` surface is now a compatibility re-export module.

Canonical ownership is split across:

- `config_io.py`
- `runtime_env.py`
- `checkpointing.py`
- `distributed.py`

Importing from `utils.training_utils` still works, but new code should prefer the
owner modules when the dependency is narrow.

## `config_io.py`

- `load_json_config` / `save_json_config`: Read and persist JSON experiment configs.
- `allocate_run_dir`: Pick the next available run directory with `_runN` suffixes.

## `runtime_env.py`

- `set_seed`: Seed Python, NumPy, and PyTorch RNGs when provided.
- `resolve_device`: Normalize manual device configuration against a default torch device.
- `resolve_batch_size`: Resolve train/eval batch-size aliases cleanly.
- `summarize_model`: Compact parameter summary (prefers `torchinfo` when available).

## `checkpointing.py`

- `safe_torch_load`: Safe deserialization helper with `weights_only=True` support when available.
- `latest_checkpoint` / `save_checkpoint`: Convenience helpers for checkpoint persistence.
- `maybe_load_checkpoint`: Generic resume helper.

## `distributed.py`

- `setup_distributed`
- `is_distributed`
- `is_main_process`

## `training_utils.py`

- Compatibility re-export surface over the four modules above.

## `evaluation_utils.py`

- `latent_shape`: Infer latent tensor dimensions from a VAE config.
- `make_grid`: Tile a batch of image tensors into a single grid (auto-expands grayscale to RGB).
- `save_image`: Persist numpy arrays to disk (creates parent directories as needed).
- `prepare_eval_batch`: Assemble an evaluation batch from a dataset on the desired device.

## Training Outputs

- `metrics.csv`: Per-epoch loss logs saved under each run directory.
- Visual probes (if enabled): fixed-batch grids saved under `<output_dir>/visuals/` (diffusion/flow) or `<output_dir>/epochs/` (VAE).
  Configure with `training.save_images`, `training.save_images_every`, and `training.visual_samples`.

## `dataset_utils.py`

- `consecutive_paths`: Generates stride-1 consecutive path groups of length `split`.
- `resolve_entry` / `split_volume_entry`: Expand directory or volume files into windowed entries.
- `build_dataset_from_config` / `build_train_val_datasets`: Resolve and instantiate datasets from config.

## `dataset_runtime.py`

- `cache_path_for_entry`: Stable cache/output path derivation for dataset entries.
- `save_tensor_cache`: Atomic tensor persistence for cache/runtime outputs.
- `iter_batches`: Small runtime batch iterator for export/sampling helpers.
- `save_output_tensor`: Save tensors under a dataset-shaped output root.

## `sampling_utils.py`

- `load_run_config`: Read `train_config.json` from a checkpoint dir.
- `resolve_checkpoint`: Pick the best/last checkpoint for a model type.
- `build_sampling_dataset`: Build a dataset for sampling with optional split override.
- `resolve_output_root`: Resolve output directory for saved tensors.
- `build_diff_map` / `save_diff_map_grid`: Shared runtime diff-map helpers.

## Tests

- Use `pytest` (or `make test`) for import smoke coverage and unit/integration checks.
- Module-level ad-hoc `_run_self_tests()` hooks were removed from production code to keep runtime modules clean and centralize validation in `tests/`.

## `model_utils/`

- `diffusion_loading.py`: Build diffusion/flow models and handle legacy checkpoint remapping.
- `diffusion_runtime.py`: Shared diffusion/flow encode/decode/visual runtime helpers.
- `diffusion_utils.py`: Compatibility re-export layer over the two modules above.
- `vae_utils.py`: Build VAEs and share encode/decode/reconstruct helpers.
