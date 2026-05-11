# `src.utils`

Utilities supporting data loading and experiment management.

## Datasets

Dataset implementations live under `src/datasets/`:
- `BaseDataset`, `LDCTDataset`, and `MNISTDataset` with full docstrings in that package.

### `build_dataset_from_config` / `build_train_val_datasets`
Config-driven dataset builders used by the training pipeline. Dataset classes are resolved via `dataset.json` in the config directory (or its parents), so each config tree can declare its own dataset implementation.

## Utility split

- `io_utils.py`: `load`, `load_image`, `load_composite` (DICOM/NumPy/Tensor/image IO).
- `dataframe_utils.py`: `lot_id` helpers for deterministic case/slice IDs.
- `indexing_utils.py`: sample-selection helpers like `select_visual_indices`.
- `utils.py`: compatibility shim re-exporting the functions above.

## `training_utils.py`

- `load_json_config` / `save_json_config`: Read and persist JSON experiment configs.
- `set_seed`: Seed Python, NumPy, and PyTorch RNGs when provided.
- `resolve_device`: Normalize manual device configuration against a default torch device.
- `summarize_model`: Compact parameter summary (prefers `torchinfo` when available).
- `allocate_run_dir`: Pick the next available run directory with `_runN` suffixes.
- `latest_checkpoint` / `save_checkpoint`: Convenience helpers for persisting checkpoints under `checkpoints/`.

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
- `iter_batches`: Iterates dataset samples in fixed-size batches.
- `save_output_tensor`: Saves tensors under an output root using the cache path structure.

## `sampling_utils.py`

- `load_run_config`: Read `train_config.json` from a checkpoint dir.
- `resolve_checkpoint`: Pick the best/last checkpoint for a model type.
- `build_sampling_dataset`: Build a dataset for sampling with optional split override.
- `resolve_output_root`: Resolve output directory for saved tensors.

## Tests

- `run_tests.py` auto-discovers `run_self_tests()` hooks and runs import smoke checks for all modules under `src/`.
- `tests/test_all_modules.py` provides pytest smoke coverage for every module and runs available `run_self_tests()` hooks.

## `model_utils/`

- `diffusion_utils.py`: Build diffusion/flow models and share encode/decode helpers for training/sampling.
- `vae_utils.py`: Build VAEs and share encode/decode/reconstruct helpers.
