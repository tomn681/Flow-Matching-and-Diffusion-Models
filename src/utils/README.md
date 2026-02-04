# `src.utils`

Utilities supporting data loading and experiment management.

## Datasets

Dataset implementations live under `src/datasets/`:
- `BaseDataset`, `LDCTDataset`, and `MNISTDataset` with full docstrings in that package.

### `build_dataset_from_config` / `build_train_val_datasets`
Config-driven dataset builders used by the training pipeline. Dataset classes are resolved via `dataset.json` in the config directory (or its parents), so each config tree can declare its own dataset implementation.

## `utils.py`

- `load(path_or_paths, id, dim)`: Unified loader for single files, directories, or lists of paths. Supports DICOM (`pydicom`), NumPy, PyTorch tensors, and standard image formats.
- `load_image` / `load_composite`: Lower-level helpers behind `load` for individual files or batched composites (with optional multiprocessing).
- `lot_id(df, case_column, number_column)`: Assigns deterministic IDs to slice windows to avoid collisions.

These helpers are used by the VAE training pipeline to build PyTorch `DataLoader`s that stream CT slices directly from disk.

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

## `model_utils/`

- `diffusion_utils.py`: Build diffusion/flow models and share encode/decode helpers for training/sampling.
- `vae_utils.py`: Build VAEs and share encode/decode/reconstruct helpers.
