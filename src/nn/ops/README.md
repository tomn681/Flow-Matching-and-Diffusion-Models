# `src.nn.ops`

Primitive operations that underpin the higher-level blocks.

## Files

- `convolution.py`
  - `ConvND`: Wrapper around `nn.Conv1d/2d/3d` selected via `spatial_dims`.
  - `ConvTransposeND`: Transposed convolution counterpart for upsampling or decoder heads.

- `pooling.py`
  - `PoolND` / `UnPoolND`: Patchify/unpatch operations that downscale or restore spatial resolution with learnable projections (used by the UNet when `pool_factor > 1`).

- `upsampling.py`
  - `UpsampleND`: Nearest-neighbour upsample followed by an optional 3×3 conv.
  - `DownsampleND`: Strided 3×3 convolution (or average pooling) for learned downsampling.

- `time_embedding.py`
  - `timestep_embedding`: Generates sinusoidal embeddings of diffusion timesteps (with optional scaling).

All functions accept `spatial_dims ∈ {1, 2, 3}` so the same code path works for signals, images, and volumes. These ops are composed inside the residual/attention blocks and model definitions.***
