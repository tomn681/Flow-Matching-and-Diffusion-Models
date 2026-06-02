# Multi-Resolution Training

Multi-resolution training uses a stepwise schedule that changes dataset resolution at specific epoch boundaries.

## Why Use It

- Faster early training at lower resolution.
- Later high-resolution refinement without separate training jobs.
- One config controls the full progressive schedule.

## Supported Architectures

Supported (fully convolutional framework models):
- `EfficientUNetND`
- `UNetDiffusersND`
- `UNet2DConditionND`
- `AutoencoderKL`
- `VQVAE`

Not supported:
- Models with fixed positional embeddings (for example DiT variants with fixed `pos_embed` grids).

## Minimal Config Example

```json
{
  "training": {
    "epochs": 20,
    "batch_size": 8,
    "multi_resolution": [
      { "start_epoch": 0, "resolution": 64 },
      { "start_epoch": 5, "resolution": 128 },
      { "start_epoch": 12, "resolution": 256 }
    ]
  }
}
```

## Important Warning

Conditioning tensors are resized in sync with target tensors at read-time.  
If you use pre-cached latents at a fixed resolution, set:

- `model.use_presaved_latents: false`

when enabling multi-resolution schedules.

## Attention Resolution Warnings

UNet-based models may log warnings when a scheduled resolution does not align with configured attention resolutions.  
If this appears:

- verify your `unet.attention_resolutions` values
- or adjust schedule resolutions to match downsample/attention structure
