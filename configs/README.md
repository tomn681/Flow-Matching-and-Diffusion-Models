# `configs/`

Configuration files live under `configs/` and drive all training, sampling, and evaluation flows.
Each config is a JSON file with (at minimum) two sections:

- `training`: training/runtime settings
- `model`: model architecture settings (must include `model_type`)

Dataset class selection is declared in a `dataset.json` file placed in the same folder
as the training config (or a parent folder). The loader walks up the directory tree
until it finds a `dataset.json`.

---

## Core schema

```json
{
  "training": { ... },
  "model": { ... }
}
```

### `model` section

Required:
- `model_type`: one of `vae`, `diffusion`, `flow_matching`

VAE models include architecture keys directly in `model` (e.g., `in_channels`, `resolution`, `z_channels`, etc.).
Diffusion/flow models define a nested UNet and scheduler:

```json
"model": {
  "model_type": "diffusion",
  "unet": { ... },
  "scheduler": { ... }
}
```

---

## Training section (common)

These keys are used by all trainers when present:

- `data_root` (string): dataset root path
- `output_dir` (string): checkpoint directory
- `seed` (int): random seed
- `num_workers` (int)
- `save_every` (int): checkpoint/save frequency (epochs)
- `save_images` (bool): enable visual probes
- `save_images_every` (int): how often to save visuals
- `visual_samples` (int): number of fixed samples to visualize
- `use_tensor_cache` / `save_tensor_cache` (bool): tensor cache behavior
- `tensor_cache_subdir` (string): cache folder name
- `norm` (bool): enable normalization
- `img_size` (int or tuple): resize target

---

## VAE training keys

Common VAE keys used by `src/pipelines/train/vae_lib.py`:

- `batch_size` (int)
- `epochs` (int)
- `learning_rate` (float)
- `weight_decay` (float)
- `recon_type` (string): `l1`, `mse`, `bce`, `bce_focal`
- `reg_type` (string): `kl` or `vq`
- `kl_weight` (float)
- `kl_anneal_steps` (int)
- `perceptual_weight` (float)
- `gan_weight` (float)
- `gan_start` (int)
- `disc_lr` (float)
- `allow_microbatching` (bool)
- `use_amp` (bool)

VAE visual probes save:
- `input.png`, `recon.png`, `gen.png` under `<output_dir>/epochs/epochXXXX/`

---

## Diffusion / Flow matching keys

Common keys for both `diffusion_lib.py` and `flow_matching_lib.py`:

- `train_batch_size` (int)
- `eval_batch_size` (int)
- `num_epochs` (int)
- `learning_rate` (float)
- `weight_decay` (float)
- `gradient_accumulation_steps` (int)
- `lr_warmup_steps` (int)
- `conditioning` (string): `concatenate` or `null`
- `channels` (int)
- `mixed_precision` (string): `no`, `fp16`, `bf16`

Visual probes save (if `save_images=true`):
- `input`, `output`, `target` grids under `<output_dir>/visuals/`

---

## Dataset selection (`dataset.json`)

Example:

```json
{
  "dataset_class": "datasets.ldct:LDCTDataset",
  "preprocess_kwargs": {
    "MIN_B": -1024,
    "MAX_B": 3072,
    "slope": 1.0,
    "intersept": -1024
  }
}
```

The loader will merge `dataset.json` values into `training` when building datasets.

---

## Sampling / Encoding / Decoding

All sampling tools use the run directory that contains `train_config.json`.

```bash
python run_model.py --ckpt_dir checkpoints/my_run --mode sample
python run_model.py --ckpt_dir checkpoints/my_run --mode encode
python run_model.py --ckpt_dir checkpoints/my_run --mode decode
python run_model.py --ckpt_dir checkpoints/my_run --mode evaluate
```

Add `--data_txt <file>` to override the split file (e.g., to sample a custom list).
