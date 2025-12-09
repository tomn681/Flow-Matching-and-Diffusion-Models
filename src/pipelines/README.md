# `src.pipelines`

Executable training and evaluation entry points. Training is now organized under `src/pipelines/train/` and dispatched via `python -m src.train` or `python train.py --config <json>`. A lightweight library-style VAE trainer is available via `src.pipelines.train.vae_lib.train(train_ds, json_path, val_ds)`. Optional LR schedulers can be defined in `training.scheduler` (StepLR, CosineAnnealingLR, ExponentialLR).

## `train/vae_lib.py`

Library-style trainer that consumes `(dataset, json_path)` (or via `python train.py --config <json>`) and builds models through `build_from_json`.

Features:
- Losses: `recon_type` in `{l1,mse,bce}`; LPIPS via `perceptual_weight>0`; GAN via `gan_weight>0`; KL vs VQ via `latent_type`/`reg_type`; KL annealing via `kl_anneal_steps`.
- Devices: main model uses `training.device`; perceptual/decoder/discriminator can be placed on separate devices via `perceptual_device` / `disc_device`.
- Micro-batching: automatic fallback on OOM (opt out with `allow_microbatching=false`); works with AMP (`use_amp`) and gradient accumulation.
- Checkpointing: keeps `vae_best.pt` and `vae_last.pt` (written every `save_every` and at the end), plus fixed-sample grids under `<output-dir>/samples/recon|gen` using the same 20 examples for comparability.
- Resume: set `training.resume` to a checkpoint path (or `true` to pick the latest in `output_dir`; `"None"` disables).

Example:
```bash
python train.py --config configs/vae.json
```
- **Dataset loader**: `DefaultDataset` (LDCT) or `MNISTDataset` via `training.dataset=mnist`; both emit tensors in `[0,1]` that the trainer rescales to `[-1,1]` and auto-resizes to the configured resolution.

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
