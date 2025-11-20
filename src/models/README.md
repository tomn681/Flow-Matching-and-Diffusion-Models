# `src.models`

High-level model definitions built from the reusable blocks under `src.nn`.

## Modules

- `vae/` – Implements a Stable-Diffusion style AutoencoderKL with configurable dimensionality. The README in that folder documents the complete encoder/decoder topology.
- `unet/` – Houses the `EfficientUNetND`, a time-conditioned UNet used for diffusion models and other denoising tasks.

`src/models/__init__.py` exposes:

```python
from src.models import vae, unet
from src.models.vae import AutoencoderKL
```

This allows a single import (`from src.models import AutoencoderKL`) in training code while still providing namespace access to `vae` and `unet` internals.

Training entry points live under `src/pipelines/train/` and are dispatched via `python -m src.train --trainer <name> --config <json> --data-root <path>`.
