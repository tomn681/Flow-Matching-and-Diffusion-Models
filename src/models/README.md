# `src.models`

High-level model definitions built from the reusable blocks under `src.nn`.

## Modules

- `vae/` – Implements Stable-Diffusion-style autoencoders: KL (`AutoencoderKL`) and configurable VQ (`VQVAE`). Paper-level VQ variants are selected in config through `quantizer_type` and `discriminator_type`.
- `unet/` – Houses the `EfficientUNetND`, a time-conditioned UNet used for diffusion models and denoising tasks.

`src/models/__init__.py` exposes the primary classes so you can import directly:

```python
from src.models import AutoencoderKL, VQVAE
```

It also exposes `build_from_json`, which currently builds VAE models from JSON configs via `VAEFactory`.

Training entry points live under `src/pipelines/train/`. The repo-root `train.py` dispatches by `model_type`, while `python -m src.train` exposes the lower-level trainer-specific CLI with explicit overrides.
