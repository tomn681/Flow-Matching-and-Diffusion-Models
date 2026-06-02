# `src.datasets`

Dataset implementations used by training and sampling.

## `base.py`

- `BaseDataset`: Generic image dataset with optional conditioning and tensor caching.
- Supports `split_file` override to point at a manual split text file.
- Defines the canonical dataset image contract through `to_image(...)` and `from_image(...)`.

## `ldct.py`

- `LDCTDataset`: LDCT/SDCT dataset that expands windowed slices and applies HU normalization.
- Adds cache-aware split metadata used for per-slice tensor caching and saving.
- Clamps normalized CT slices into canonical image space `[0, 1]` and can invert back to the configured HU window with `from_image(...)`.

## `mnist.py`

- `MNISTDataset`: Lightweight MNIST loader for smoke tests or minimal training runs.
- Exposes `to_image(...)` / `from_image(...)` for `[0, 1] <-> [0, 255]` conversion.

## `medical3d.py`

- `Medical3DDataset`: Generic 3D medical volume dataset for volumetric training.
- Loads `.npy`/`.npz`/`.pt`/`.pth` volumes and optionally NIfTI volumes when `nibabel` is installed.
- Returns volume tensors shaped as `(C, D, H, W)` and reuses the standard cache/conditioning contract.

## `video.py`

- `VideoDataset`: Groups per-frame annotations into sliding temporal clips.
- Returns clip tensors shaped as `(C, T, H, W)` for target and optional conditioning streams.
- Keeps clip caching unique by storing split metadata per temporal window.
