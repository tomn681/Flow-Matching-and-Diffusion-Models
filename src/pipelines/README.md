# `src.pipelines`

Executable training and evaluation entry points. Training is now organized under `src/pipelines/train/` and dispatched via `python -m src.train` or `python train.py --config <json>`. A lightweight library-style VAE trainer is available via `src.pipelines.train.vae_lib.train(train_ds, json_path, val_ds)`. Optional LR schedulers can be defined in `training.scheduler` (StepLR, CosineAnnealingLR, ExponentialLR).

## `train/vae_lib.py`

Library-style trainer that consumes `(dataset, json_path)` (or via `python train.py --config <json>`) and builds models through `build_from_json`. Use the JSON presets under `configs/` for ready-made settings. Losses: `recon_type` supports `l1`, `mse`, or `bce`; add LPIPS with `perceptual_weight>0`; enable GAN with `gan_weight>0`; select KL or VQ via `latent_type`/`reg_type`. Optional schedulers live under `training.scheduler`.

Example:
```bash
python train.py --config configs/vae.json
```
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
