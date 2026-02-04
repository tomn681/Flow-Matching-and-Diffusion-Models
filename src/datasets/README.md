# `src.datasets`

Dataset implementations used by training and sampling.

## `base.py`

- `BaseDataset`: Generic image dataset with optional conditioning and tensor caching.
- Supports `split_file` override to point at a manual split text file.

## `ldct.py`

- `LDCTDataset`: LDCT/SDCT dataset that expands windowed slices and applies HU normalization.
- Adds cache-aware split metadata used for per-slice tensor caching and saving.

## `mnist.py`

- `MNISTDataset`: Lightweight MNIST loader for smoke tests or minimal training runs.
