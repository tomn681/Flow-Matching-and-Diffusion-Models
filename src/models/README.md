# `src.models`

High-level model definitions built from the reusable blocks under `src.nn`.

## Modules

- `vae/` – Implements Stable-Diffusion-style autoencoders: KL (`AutoencoderKL`), EMA VQ (`VQVAE`), and Magvit VQ (`MagvitVQVAE`).
- `unet/` – Houses:
  - `BaseUNetND` shared forward scaffold
  - `EfficientUNetND` (generic ND UNet)
  - `UNetDiffusersND` (Diffusers-compatible ND UNet)
- `utils/` – Model-level utilities such as checkpoint/state-dict merging helpers.

`src/models/__init__.py` exposes the primary classes so you can import directly:

```python
from src.models import AutoencoderKL, VQVAE, merge_models
```

## Canonical Construction

The canonical model-construction API is:

```python
from models import ModelFactory

model = ModelFactory.build(config)
```

Use this when you already have a validated config dictionary in memory.

Model classes themselves are initialization-only and do not define a parallel
`from_config()` story. Config loading belongs to:

- trainer `from_config(...)` helpers for training orchestration
- sampler/facade helpers for runtime checkpoint loading
- small compatibility wrappers that delegate back to `ModelFactory.build(...)`

## Training / Runtime Ownership

- Canonical training entrypoints live in `src.training` and the supported root `train.py` CLI.
- Canonical runtime sampler ownership lives in `src.sampling` and the supported root `run_model.py` CLI.
- `src/pipelines/train/*_lib.py` and `src/models/generators/*factory.py` are compatibility surfaces, not the primary API story.
