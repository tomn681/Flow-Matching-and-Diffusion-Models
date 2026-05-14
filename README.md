# Flow Matching and Diffusion Models

_Flow Matching and Diffusion Models_ is an independent research codebase for training latent autoencoders, diffusion pipelines, and flow-matching pipelines on Low-Dose CT (LDCT) data. It implements KL and VQ autoencoders, with the VQ recipe selected from config (original VQ-VAE, EMA-style tokenizer, or MAGVIT-style discriminator) alongside reusable neural building blocks that generalise to 1D/2D/3D workloads.

## Project Layout

- `src/` – Python package with reusable modules.
  - `nn/` – Core neural primitives (residual blocks, attention, pooling/upsampling, time embeddings).
  - `models/` – High-level model assemblies (AutoencoderKL, EfficientUNetND, UNetDiffusersND).
  - `datasets/` – Dataset implementations (BaseDataset, LDCTDataset, MNISTDataset).
  - `pipelines/` – Train/eval entry points plus sampling/encoding/decoding utilities.
  - `utils/` – Dataset loaders and helper utilities.
- `configs/` – Training configs and dataset selectors (`dataset.json`).
- `checkpoints/` – Output directory for model checkpoints, configs, and per-epoch metric logs (ignored by git).
- `run_tests.py` – Module self-test + import-smoke runner.
- `tests/` – Pytest-based module smoke tests.

## Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Train a model:

```
python train.py --config configs/LDCT/LDCT_autoencoder_kl_test.json
```

## Configs

All configs are JSON and contain:

- `training`: runtime/training settings
- `model`: architecture settings (must include `model_type`)

See [configs/README.md](/Users/delas/Documents/LDCT/Flow-Matching-and-Diffusion-Models/configs/README.md) for the parameter reference and canonical config families.

## Training & Validation

All training is driven by JSON configs (`configs/*.json`) and a single dispatcher:

```
python train.py --config path/to/config.json [--resume optional_ckpt]
```

- **VAEs** (`configs/autoencoder_kl*.json`, `configs/fmboost_autoencoder_kl.json`, `configs/ldm_autoencoder_kl.json`, `configs/vqvae*.json`, plus dataset-specific variants under `configs/LDCT/` and `configs/MNIST/`): configs expose `training` + `model` sections. Features include automatic micro-batching on OOM, optional perceptual/GAN losses, configurable schedulers, and per-epoch validation (built from the test split) when `train.py` instantiates datasets via `build_train_val_datasets`.
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

All sampling tools use the run directory that contains `train_config.json`.

```
python run_model.py --ckpt_dir <run_dir> --mode sample
python run_model.py --ckpt_dir <run_dir> --mode encode
python run_model.py --ckpt_dir <run_dir> --mode decode
python run_model.py --ckpt_dir <run_dir> --mode evaluate
```

Options:
- `--ckpt_dir <path>`: checkpoint/run directory (required).
- `--mode {sample,encode,decode,evaluate,build_tensor_cache,debug_compare}`: workflow to run.
- `--data_txt <path>`: override split file (`train.txt`/`test.txt` style).
- `--save`: write generated outputs to disk.
- `--output_dir <path>`: output root override (default depends on mode).
- `--batch_size <int>`: processing batch size.
- `--device <str>`: torch device (for example: `cuda`, `cpu`, `cuda:0`).
- `--seed <int>`: RNG seed.
- `--timestep <int>`: encode timestep override (encode mode).
- `--num_samples <int>`: randomly sample only N dataset items.
- `--num_inference_steps <int>`: scheduler inference-step override.
- `--start_step <int>`: start denoising from train-timestep `N` down to `0`.
- `--last_n_steps <int>`: run only the last N denoising steps.
- `--scheduler <name>`: runtime scheduler override:
  - `ddpm`, `ddim`, `dpmsolver1`, `dpmsolver2`, `dpmsolver++`, `dpmsolversde`, `unipc`, `flowmatch`
- `--save_input`: with `--save`, also save input/target tensors.
- `--save_conditioning`: with `--save`, also save conditioning tensors.
- `--save_tensor_cache`: force writing tensor cache files at runtime (without editing `train_config.json`).

Notes:
- `build_tensor_cache` writes cache files when either:
  - config has `training.save_tensor_cache=true`, or
  - CLI includes `--save_tensor_cache`.
- Outputs (if enabled) are saved with the same directory structure as the input data.
- Evaluation metrics location:
  - with `--output_dir` in `evaluate` mode: a unique experiment subfolder is created under `--output_dir`, and metrics/artifacts are written there;
  - without `--output_dir`: metrics are written under `--ckpt_dir`.

## Visual Monitoring

Visual probes are saved from a fixed batch for training inspection:
- VAE: `input.png`, `recon.png`, `gen.png` under `<output_dir>/epochs/epochXXXX/`
- Diffusion/Flow: `input/output/target` grids under `<output_dir>/visuals/`

Control with:
- `training.save_images` (bool)
- `training.save_images_every` (int)
- `training.visual_samples` (int)

## Metrics Logging

Each run writes `metrics.csv` under the run directory for easy plotting.

## Testing

Run the test harness:

```
python run_tests.py
```

Run pytest smoke tests:

```
python -m pytest -q tests/test_all_modules.py
```

## Library Usage

Build models from JSON:

```python
from models import build_from_json
vae = build_from_json("configs/autoencoder_kl.json")
```

Programmatic training:

```python
from pipelines.train import train_vae, train_flow_matching, train_diffusion
train_vae(train_dataset, "configs/autoencoder_kl.json", val_dataset)
train_flow_matching(train_dataset, "configs/flow_matching/ldct_flow_matching.json")
```

Programmatic sampling/encoding/decoding/evaluation:

```python
from pipelines.samplers.handlers import VAEHandler, DiffusionHandler, FlowMatchingHandler

handler = VAEHandler(ckpt_dir="checkpoints/ldct_vae_test_run1")
handler.sample()
handler.decode()
handler.evaluate()

handler = DiffusionHandler(ckpt_dir="checkpoints/ldct_ddpm_test_run1", save=True)
handler.encode()
handler.decode()
handler.evaluate()
```

Shared utilities (`utils`) expose dataset builders, config IO, checkpoint helpers, distributed setup, and evaluation tools. Reuse them in external scripts to keep behaviour consistent.

See the README inside each subdirectory for finer-grained documentation of available modules and configuration options.
