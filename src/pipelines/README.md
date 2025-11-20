# `src.pipelines`

Executable training and evaluation entry points. Training is now organized under `src/pipelines/train/` and dispatched via `python -m src.train`.

## `train/vae.py`

Callable trainer that wraps `AutoencoderKL` and `DefaultDataset`. It is invoked through the generic dispatcher:

```
python -m src.train \
  --trainer vae \
  --config configs/vae.json \
  --data-root /path/to/data \
  [--epochs 50] [--batch-size 2] [--img-size 128] [--in-channels 1] [--out-channels 1]
```

- **Configuration via JSON**: See `configs/vae.json` (full/default) and `configs/vae_small.json` (lighter, 1 resblock) for ready-made settings. The dispatcher writes the resolved config to `<output_dir>/train_config.json` for reproducibility.
- **Minimal overrides**: Only a handful of CLI overrides are supported for quick experiments (`epochs`, `batch_size`, `img_size`, `in_channels`, `out_channels`, `perceptual_device`, `gan_device`, `micro_batch_size`). All other hyperparameters come from the JSON.
- **Losses**: Supports perceptual/MSE/BCE reconstruction, KL or VQ regularisation with annealing, and optional GAN hinge loss.
- **Dataset loader**: `DefaultDataset` yields `target` tensors (scaled from `[0,1]` to `[-1,1]`); automatic resize if shapes differ from the configured resolution.
- **Checkpointing + metrics**: Saves periodic and best checkpoints; logs per-epoch metrics to `metrics.jsonl`.

## `samplers/vae.py`

VAE sampler invoked via the generic dispatcher:

```
python -m src.sample \
  --sampler vae \
  --checkpoints-root checkpoints \
  --run-name vae_run1 \
  [--samples 25] [--checkpoint best|last|both]
```

- If `--run-name` is omitted, the most recently updated directory inside `checkpoints/` is used.
- Reads the saved `train_config.json`, loads checkpoints, reconstructs a grid of validation slices, and decodes a grid of random latent samples.
- PNGs (`vae_recon_grid_<label>.png`, `vae_generated_grid_<label>.png`) are written into the run directory. Use `--samples` to change grid size (must be a perfect square). Pass `--checkpoint both` to export grids for both best and last checkpoints.
