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
- Automatic micro-batching on OOM (opt out via `allow_microbatching=false`); mixed precision via `use_amp` (default false).
- Run folders auto-increment: `output_dir` becomes `<stem>_runN` unless resuming.
- Checkpointing every epoch: always keeps `vae_best.pt` and `vae_last.pt`; `save_every` also writes `epochs/epoch####/epoch.pt` plus recon/gen grids (fixed 20 samples for comparability). Final epoch saves again.
- Resume via CLI `--resume` (uses latest checkpoint in the run if no path is given).
- Inputs are kept in `[0,1]` (MNIST loader downloads/resizes to 32×32 and normalises via `/255`); evaluation/sample grids apply sigmoid to logits for display.
- Loss composition: `recon_type` in `{l1,mse,bce,bce_focal}`; `perceptual_weight>0` adds LPIPS; `gan_weight>0` enables PatchGAN; KL annealing via `kl_anneal_steps`; KL vs VQ picked by `latent_type`/`reg_type` (codebook loss is ignored when not in VQ mode).
- Model summary: compact list of conv/linear/attention/pool layers with parameter counts is printed before training alongside a cyan data line (now includes epochs).

## Library usage (JSON-driven)

- Build a model from JSON: `from models import build_from_json; model = build_from_json("configs/vae.json")` (supports KL, VQ, and Magvit VQ via `latent_type`). The VAE factory lives at `models/generators/vaefactory.py`.
- Train with a dataset: `from pipelines.train.vae_lib import train; train(train_dataset, "configs/vae.json", val_dataset)`
- Or run from the repo root: `python train.py --config configs/vae.json` (expects `training.data_root` in the JSON).
- Optional: add a scheduler under `training.scheduler` (e.g., `{"name": "StepLR", "params": {"step_size": 10, "gamma": 0.5}}`). If omitted, no scheduler is used. Passing `val_dataset` (or letting `train.py` build one) enables per-epoch validation logging.

Only VAEs declared in the existing JSONs plus the VQ-VAE are supported. Prefer the modular `models/vae` components. Configs support `down_channels` (absolute widths) as a preferred alternative to `ch_mult`, `norm_groups` to override GroupNorm grouping, and `codebook_size` (used only by VQ variants; ignored for KL).

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
