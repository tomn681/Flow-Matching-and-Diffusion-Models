# `src.utils`

Utilities supporting data loading and experiment management.

## `dataset.py`

### `DefaultDataset`
- Reads `train.txt` / `test.txt` metadata tables (tab-separated) from a dataset root. Each row specifies case IDs plus paths to standard-dose (SDCT) and low-dose (LDCT) volumes. Directories are expanded into overlapping slice windows via `n_slice_split`.
- Returns dictionaries with:
  - `target`: Standard-dose tensor (`torch.float32`) after normalisation to `[0, 1]` (or `[-1, 1]` downstream).
  - `image`: Optional low-dose tensor (only populated when `train=False` or `diff=False`).
  - `img_id`, `img_path`, `img_size`: Metadata describing the source slice.
- Preprocessing:
  - Converts DICOM voxels to Hounsfield units using slope/intercept.
  - Resizes volumes to `img_size × img_size` (if provided) using `skimage.transform.resize`.
  - Normalises intensities and casts to the requested data type (`np.float32` by default).
- Logging: reports dataset cardinality when instantiated.

### `CombinationDataset`
Extends `DefaultDataset` to also load sinogram data (`SDRAW`, `LDRAW`) for hybrid reconstruction tasks.

## `utils.py`

- `load(path_or_paths, id, dim)`: Unified loader for single files, directories, or lists of paths. Supports DICOM (`pydicom`), NumPy, PyTorch tensors, and standard image formats.
- `n_slice_split(directory, split)`: Generates stride-1 windows of length `split` for multi-slice stacks.
- `lot_id(df, case_column, number_column)`: Assigns deterministic IDs to slice windows to avoid collisions.

These helpers are used by the VAE training pipeline to build PyTorch `DataLoader`s that stream CT slices directly from disk.***
