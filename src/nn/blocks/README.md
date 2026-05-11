# `src.nn.blocks`

Reusable higher-level blocks used across models.

## `common.py`

- `zero_module`: shared zero-initializer used by residual and attention blocks.

## `residual.py`

### `ResBlockND`
- Residual block that works in 1D/2D/3D.
- Pipeline: `GroupNorm → SiLU → Conv` → optional FiLM/scale-shift → `GroupNorm → SiLU → Dropout → Conv`.
- Skip path is identity when `channels == out_channels`, otherwise a 1×1 or 3×3 conv depending on `use_conv`.
- Supports conditioning through `emb_channels`; the embedding is projected via an MLP and added (or scale/shifted) before the second convolution.
- Used in both the VAE and UNet encoders/decoders.

## `attention.py`

- `SpatialSelfAttention`: Multi-head self-attention applied over flattened spatial tokens. Supports both efficient PyTorch SDPA and a fallback implementation. Can optionally switch to linear attention.
- `QKVAttention` / `LinearQKVAttention`: Internal helpers implementing the attention mechanisms.
- All attention modules preserve the original tensor shape `(B, C, *spatial)`.

## `timestep.py`

- Defines the `TimestepBlock` protocol and wrappers enabling modules to accept diffusion timestep embeddings.
- `TimestepEmbedSequential` integrates with PyTorch `nn.Sequential` to forward embeddings to child blocks that implement `TimestepBlock`.

## `legacy_unet.py`

- Diffusers-compat UNet blocks used by `UNetDiffusersND`:
  - `DownBlock2DCompat`
  - `UpBlock2DCompat`
  - `UNetMidBlock2DCompat`

These blocks are composed in `src.models` to build the AutoencoderKL and EfficientUNetND networks.
