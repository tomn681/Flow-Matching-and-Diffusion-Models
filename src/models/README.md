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

Training entry points live under `src/pipelines/train/` and are dispatched via `python train.py --config <json>`.
