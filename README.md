# Flow Matching and Diffusion Models

_Flow Matching and Diffusion Models_ is an independent research codebase for training latent autoencoders, diffusion pipelines, and flow-matching pipelines on Low-Dose CT (LDCT) data. It implements multiple VAE variants (Stable Diffusion VAE, SD2 VAE, VQVAE/VQGAN, FMBoost VAE, MagViT VAE) alongside reusable neural building blocks that generalise to 1D/2D/3D workloads, allowing pixel and latent space image generation.

## Project Layout

- `src/` – Python package with reusable modules.
  - `nn/` – Core neural primitives (residual blocks, attention, pooling/upsampling, time embeddings).
  - `models/` – High-level model assemblies (AutoencoderKL, EfficientUNetND).
  - `pipelines/` – Train/eval entry points dispatched via `python -m src.train` (training) and `python -m src.sample` (sampling).
  - `utils/` – Dataset loaders and helper utilities.
- `checkpoints/` – Output directory for model checkpoints, configs, and per-epoch metric logs (ignored by git).
- `run_tests.py` – Placeholder pytest runner.

## Training & Validation

All training is driven by JSON configs (`configs/*.json`) and a single dispatcher:

```
python train.py --config path/to/config.json [--resume optional_ckpt]
```

- **VAEs** (`configs/vae*.json`, `configs/LDCT/LDCT_*vae*.json`): config exposes `training` + `vae` sections. Features include automatic micro-batching on OOM, optional perceptual/GAN losses, configurable schedulers, and per-epoch validation (built from the test split) when `train.py` instantiates datasets via `build_train_val_datasets`.
- **Flow matching** (`configs/flow_matching/*.json`) and **diffusion/DDPM** (`configs/diffusion/*.json`): share the same `training` section (dataset root, batch sizes, cache flags) plus a model-specific block describing the Diffusers UNet and scheduler. Conditioning modes (“concatenate” LDCT, or unconditional), cosine warmup, gradient accumulation, and mixed precision are all JSON-driven.
- Validation: whenever `training.load_ldct`/`training.dataset` provide a test split, `train.py` constructs both train/val datasets so every trainer can log validation loss. Custom validation datasets can also be passed manually when calling the trainers as libraries.

### Distributed / Multi-GPU

Single GPU: run as usual (`python train.py ...`).  
Multi GPU: launch through `torchrun` (or a compatible launcher) so `WORLD_SIZE`/`LOCAL_RANK` are set:

```
torchrun --nproc_per_node=2 train.py --config configs/flow_matching/ldct_flow_matching.json
```

Both flow-matching and diffusion trainers shard the dataset via `DistributedSampler`, reduce metrics across ranks, and ensure only rank 0 writes checkpoints/samples. VAEs remain single-process but still support the `manual_device` override plus micro-batching.

## Evaluation & Sampling

- **VAE sampling**: `python -m src.sample --sampler vae --checkpoints-root checkpoints --run-name <run>` renders recon/gen grids for `best` and/or `last` checkpoints saved during training. The sampler reuses the saved `train_config.json`, loads checkpoints, and writes PNGs in the run directory.
- **Flow matching / Diffusion sampling**: during training, per-epoch sample grids are written into `<output_dir>/samples/epoch####.png`. Dedicated standalone samplers will reuse the shared `pipelines.utils` helpers; until then, re-run the trainer in eval mode or call `sample_with_scheduler` manually.
- **Evaluation batches**: use `utils.prepare_eval_batch(dataset, n, device)` to assemble fixed batches for visualisations or metrics.

## Testing

- `run_tests.py` is a placeholder entrypoint for future regression tests (pytest compatible).
- Model-specific tests can be added under `tests/` (not yet committed) and invoke dataset factories, model builders, or trainers as needed.

## Library Usage

- Build models from JSON:
  ```python
  from models import build_from_json
  vae = build_from_json("configs/vae.json")
  ```
- Programmatic training:
  ```python
  from pipelines.train import train_vae, train_flow_matching, train_diffusion
  train_vae(train_dataset, "configs/vae.json", val_dataset)
  train_flow_matching(train_dataset, "configs/flow_matching/ldct_flow_matching.json")
  ```
- Shared utilities (`utils`) expose dataset builders, config IO, checkpoint helpers, distributed setup, and evaluation tools. Reuse them in external scripts to keep behaviour consistent.

See the README inside each subdirectory for finer-grained documentation of available modules and configuration options.
