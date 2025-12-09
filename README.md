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

Key behaviours (see `configs/vae*.json` for presets):
- Automatic micro-batching on OOM (opt out via `allow_microbatching=false`); mixed precision via `use_amp`.
- Checkpointing: always keeps `vae_best.pt` (best val/train loss) and `vae_last.pt` (last/save_every), plus matching sample grids under `<output-dir>/samples/recon|gen` using a fixed set of 20 examples for comparability.
- Resume training via `training.resume` (path or `"None"` to disable).
- Inputs are normalised to `[-1, 1]`; spatial mismatches are interpolated to the configured resolution.
- Loss composition: `recon_type` in `{l1,mse,bce}`; `perceptual_weight>0` adds LPIPS; `gan_weight>0` enables PatchGAN; KL annealing via `kl_anneal_steps`; KL vs VQ picked by `latent_type`/`reg_type`.

## Library usage (JSON-driven)

- Build a model from JSON: `from models import build_from_json; model = build_from_json("configs/vae.json")` (supports KL, VQ, and Magvit VQ via `latent_type`). The VAE factory lives at `models/generators/vaefactory.py`.
- Train with a dataset: `from pipelines.train.vae_lib import train; train(train_dataset, "configs/vae.json", val_dataset)`
- Or run from the repo root: `python train.py --config configs/vae.json` (expects `training.data_root` in the JSON).
- Optional: add a scheduler under `training.scheduler` (e.g., `{"name": "StepLR", "params": {"step_size": 10, "gamma": 0.5}}`). If omitted, no scheduler is used. Passing `val_dataset` (or letting `train.py` build one) enables per-epoch validation logging.

Only VAEs declared in the existing JSONs plus the VQ-VAE are supported. Prefer the modular `models/vae` components.

## Sampling from the VAE

```
python -m src.sample \
  --sampler vae \
  --checkpoints-root checkpoints \
  --run-name vae_run1 \
  [--samples 25] [--checkpoint best|last|both]
```

Loads the saved `train_config.json`, restores checkpoints, writes grids of reconstructions and random generations into the run directory (suffixes `_best` / `_last` when sampling multiple checkpoints). Omit `--run-name` to use the latest run in `checkpoints/`. Grids follow the same [-1,1] normalisation and use the saved sample count.

See the README inside each subdirectory for detailed documentation of the provided modules.
