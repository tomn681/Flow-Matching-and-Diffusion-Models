# `src.nn`

Dimension-agnostic neural building blocks used by the VAE and UNet models.

## Submodules

- `blocks/` – Residual blocks, attention layers, and timestep-aware wrappers.
- `ops/` – Primitive ops: convolutions, pooling, up/down-sampling, time embeddings.
- `modules/` – Higher-level reusable modules such as VAE parts, LoRA adapters, and temporal attention.

Key design principles:

- Every operator accepts a `spatial_dims` argument (1/2/3) so the same code works for signals, images, and volumes.
- Residual blocks optionally accept conditioning embeddings (FiLM/scale-shift).
- Attention layers flatten spatial dimensions internally, enabling the same implementation to run at any dimensionality.

Inspect each subdirectory README for detailed documentation of the contained utilities.
