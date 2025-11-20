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
- **Minimal overrides**: Only a handful of CLI overrides are supported for quick experiments (`epochs`, `batch_size`, `img_size`, `in_channels`, `out_channels`). All other hyperparameters come from the JSON.
- **Losses**: Supports perceptual/MSE/BCE reconstruction, KL or VQ regularisation with annealing, and optional GAN hinge loss.
- **Dataset loader**: `DefaultDataset` yields `target` tensors (scaled from `[0,1]` to `[-1,1]`); automatic resize if shapes differ from the configured resolution.
- **Checkpointing + metrics**: Saves periodic and best checkpoints; logs per-epoch metrics to `metrics.jsonl`.

## `sample_vae.py`

Produces qualitative visualisations from a trained VAE run.

```
python -m src.pipelines.sample_vae --run-name vae_run1
```

- If `--run-name` is omitted, the most recently updated directory inside `checkpoints/` is used.
- The script reads the saved `train_config.json`, loads the best checkpoint, reconstructs a 5×5 grid of validation slices, and decodes a 5×5 grid of random latent samples.
- PNGs (`vae_recon_grid.png`, `vae_generated_grid.png`) are written into the run directory. Use `--samples` to change the grid size (must remain a perfect square). Pass `--checkpoint both` to export grids for both the best and latest checkpoints (suffixes `_best` and `_last`).
