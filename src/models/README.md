# `src.models`

High-level model definitions built from the reusable blocks under `src.nn`.

## Modules

- `vae/` – Implements Stable-Diffusion-style autoencoders: KL (`AutoencoderKL`), EMA VQ (`VQVAE`), and Magvit VQ (`MagvitVQVAE`).
- `unet/` – Houses the `EfficientUNetND`, a time-conditioned UNet used for diffusion models and denoising tasks.

`src/models/__init__.py` exposes the primary classes so you can import directly:

```python
from src.models import AutoencoderKL, VQVAE, MagvitVQVAE
```

Training entry points live under `src/pipelines/train/` and are dispatched via `python -m src.train --trainer <name> --config <json> --data-root <path>`.
