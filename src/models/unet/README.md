# `src.models.unet`

Home of `EfficientUNetND`, a diffusion-ready UNet that operates in 1D/2D/3D with optional attention and timestep conditioning.

## Architecture Overview

### Encoder (`input_blocks`)

1. **Stem:** `ConvND(spatial_dims, in_channels, model_channels, 3×3)` followed by residual blocks.
2. **Levels:** For each multiplier in `channel_mult`:
   - Run `num_res_blocks` residual blocks (`ResBlockND`) at the current width. Each block receives the global timestep embedding and optionally applies scale-shift norm.
   - Insert `SpatialSelfAttention` after a block when the running downsample factor (`ds`) matches an entry in `attention_resolutions`.
   - Between levels (except the last), append `DownsampleND` (learned if `conv_resample=True`).
3. A list of intermediate activations is stored for skip connections in the decoder.

### Bottleneck (`middle_block`)

`ResBlockND → SpatialSelfAttention → ResBlockND`, all at the deepest channel width.

### Decoder (`output_blocks`)

Iterates through `channel_mult` in reverse:

- Each level processes `num_res_blocks + 1` blocks (the final block prepares for upsampling).
- Skip connections from the encoder are concatenated channel-wise before every block.
- Attention is inserted when the current downsample factor matches `attention_resolutions`.
- After the last block of each non-final level, `UpsampleND` doubles the spatial resolution.

### Output Head

`GroupNorm → SiLU → ConvND` produces the `out_channels`. If `pool_factor > 1`, the model performs an initial patchify via `PoolND` and restores the resolution with `UnPoolND` at the end.

## Time Conditioning

- `timestep_embedding` produces sinusoidal embeddings of the diffusion timestep.
- A small MLP inflates the embedding to `4 * model_channels`, which is fed into every `ResBlockND`.

## Key Features

- Dimensionality-agnostic: works with 1D signals, 2D images, or 3D volumes.
- Optional linear attention for memory efficiency (`use_linear_attn=True`).
- Scale/shift norm conditioning enables FiLM-style modulation (enabled by default).
- Supports auxiliary context concatenation via the `context` argument in `forward`.

## Example

```python
model = EfficientUNetND(
    spatial_dims=2,
    in_channels=4,          # e.g., x + conditioning
    model_channels=256,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=(1, 2, 4),
    channel_mult=(1, 2, 4, 4),
    dropout=0.0,
)

noise_pred = model(x, timesteps)  # shape (N, 4, H, W)
```

Refer to `src/models/unet/unet.py` for further details and the included self-test harness.
