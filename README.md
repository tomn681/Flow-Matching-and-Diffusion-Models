# LDCT_v2

LDCT_v2 is a lightweight research codebase for training latent autoencoders and diffusion-ready UNets on Low-Dose CT (LDCT) data. The repository mirrors the Stable Diffusion VAE defaults while exposing reusable neural building blocks that generalise to 1D/2D/3D workloads.

## Project Layout

- `src/` – Python package with reusable modules.
  - `nn/` – Core neural primitives (residual blocks, attention, pooling/upsampling, time embeddings).
  - `models/` – High-level model assemblies (AutoencoderKL, EfficientUNetND).
  - `pipelines/` – Train/eval entry points dispatched via `python -m src.train` (training) and `python -m src.sample` (sampling).
  - `utils/` – Dataset loaders and helper utilities.
- `checkpoints/` – Output directory for model checkpoints, configs, and per-epoch metric logs (ignored by git).
- `run_tests.py` – Placeholder pytest runner.

## Training the VAE

```
python -m src.train \
  --trainer vae \
  --config configs/vae.json \
  --data-root /path/to/DefaultDataset \
  [--epochs 100] [--batch-size 4] [--img-size 256]
```

Key behaviours (see `configs/vae.json` or `configs/vae_small.json` for presets):
- Logs per-epoch train/val metrics to `<output-dir>/metrics.jsonl` and stores the best checkpoint as `vae_best_epochXXXX.pt`.
- Inputs are normalised to `[-1, 1]`. Any spatial mismatch with the model resolution is handled automatically by interpolation (warned once per loop).
- Mixed precision can be toggled with `--use-amp` to reduce VRAM use.
- Loss composition is configurable: `recon_type` supports `l1`, `mse`, or `bce` (pixel-domain base); set `perceptual_weight>0` to add LPIPS on top of the recon term (e.g., `l1+LPIPS`), and `gan_weight>0` to include a PatchGAN hinge loss. KL or VQ regularisation is selected via `latent_type`/`reg_type`.

## Sampling from the VAE

```
python -m src.sample \
  --sampler vae \
  --checkpoints-root checkpoints \
  --run-name vae_run1 \
  [--samples 25] [--checkpoint best|last|both]
```

Loads the saved `train_config.json`, restores checkpoints, writes grids of reconstructions and random generations into the run directory (suffixes `_best` / `_last` when sampling multiple checkpoints). Omit `--run-name` to use the latest run in `checkpoints/`.

See the README inside each subdirectory for detailed documentation of the provided modules.
