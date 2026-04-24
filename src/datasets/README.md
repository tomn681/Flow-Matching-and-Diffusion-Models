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
