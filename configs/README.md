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
For VQ models, recipe-level choices should remain in config, especially:
- `quantizer_type`: `classic` or `ema`
- `discriminator_type`: `patchgan` or `magvit`

Diffusion/flow models define a nested UNet and scheduler:

```json
"model": {
  "model_type": "diffusion",
  "unet": { ... },
  "scheduler": { ... }
}
```

---

## Parameter Reference

This section documents the parameters currently consumed by the codebase. It is
organized by where the key is read, not by whether every config uses it.

### Training keys shared across multiple trainers / dataset builders

- `data_root` (string): dataset root; also forwarded into dataset constructors.
- `output_dir` (string): run directory base.
- `seed` (int): random seed.
- `num_workers` (int): dataloader worker count.
- `img_size` (int or tuple): resize target forwarded to datasets.
- `norm` (bool): forwarded to datasets that support normalization.
- `slice_count` (int): fallback alias for dataset `window_size`.
- `window_size` (int): explicit dataset window size when supported.
- `split_file` (string): optional custom split file for dataset construction.
- `use_tensor_cache` (bool): read cached tensors when present.
- `save_tensor_cache` (bool): write tensor cache files.
- `tensor_cache_subdir` (string): cache directory name under `data_root`.
- `save_images` (bool): enable visual probes.
- `save_images_every` (int): visual probe cadence.
- `save_every` (int): checkpoint cadence.
- `weight_decay` (float): optimizer weight decay.
- `learning_rate` (float): optimizer base learning rate.
- `scheduler` (object): optional LR scheduler block with `name` and `params`.

### VAE training keys (`src/pipelines/train/vae_lib.py`)

- `batch_size` (int): training and validation batch size.
- `epochs` (int): number of epochs.
- `reg_type` (string): `kl` or `vq`; used for loss bookkeeping.
- `recon_type` (string): `l1`, `mse`, `bce`, or `bce_focal`.
- `kl_weight` (float): KL loss weight.
- `kl_anneal_steps` (int): linear KL warmup in optimizer steps.
- `codebook_weight` (float): VQ loss weight; ignored outside VQ models.
- `perceptual_weight` (float): LPIPS weight.
- `gan_weight` (float): adversarial generator loss weight.
- `gan_start` (int): epoch-based GAN warmup fallback.
- `gan_start_steps` (int): step-based GAN warmup; takes precedence over `gan_start`.
- `disc_lr` (float or null): discriminator learning rate; defaults to `learning_rate`.
- `manual_device` (string or null): explicit training device.
- `perceptual_device` (string or null): device override for LPIPS.
- `disc_device` (string or null): device override for discriminator.
- `allow_microbatching` (bool): split batches on OOM and accumulate gradients.
- `use_amp` (bool): enable `torch.cuda.amp`.
- `visual_samples` (int): number of samples used in saved VAE probe grids.
- `resume` (string or null): optional checkpoint path used by the VAE trainer.

VAE probe outputs:
- `input.png`, `recon.png`, `gen.png` under `<output_dir>/epochs/epochXXXX/`

### Diffusion / flow-matching training keys

- `train_batch_size` (int): training batch size.
- `eval_batch_size` (int): evaluation / visual batch size.
- `num_epochs` (int): training epochs.
- `gradient_accumulation_steps` (int): optimizer accumulation interval.
- `lr_warmup_steps` (int): warmup steps.
- `conditioning` (string or null): conditioning mode; `concatenate` is the LDCT path.
- `mixed_precision` (string): `no`, `fp16`, or `bf16`.
- `num_train_timesteps` (int): training diffusion horizon.
- `num_inference_steps` (int): default inference sampler steps.
- `latent_norm` (string or null): latent normalization strategy.
- `channels` (int): default latent/image channel count for diffusion factory fallback.
- `save_model_epochs` (int): extra checkpoint cadence used by diffusion / flow code.

Flow / diffusion probe outputs:
- `input`, `output`, `target` grids under `<output_dir>/visuals/`

### Dataset-specific training keys

These are only consumed if the selected dataset class accepts them:

- `download` (bool): dataset download flag used by MNIST.
- `load_ldct` (bool): forwarded to LDCT-style datasets when supported.
- `dataset` (string): appears in configs but is not currently consumed by the shared builders.

### VAE model keys (`model` section)

Shared VAE architecture keys:

- `model_type` (string): must be `vae`.
- `latent_type` (string): `kl` or `vq`; selects `AutoencoderKL` vs `VQVAE`.
- `in_channels` / `out_channels` (int): input and output channels.
- `resolution` (int): nominal input resolution.
- `base_ch` (int): base channel width.
- `down_channels` (list[int] or null): explicit stage widths; overrides `ch_mult`-style defaults.
- `num_res_blocks` (int): residual blocks per stage.
- `attn_resolutions` (list[int]): resolutions/factors where self-attention is inserted.
- `z_channels` (int): latent channels used by encoder/decoder trunks.
- `embed_dim` (int): KL latent embedding width or VQ code embedding width.
- `dropout` (float): residual-block dropout.
- `use_attention` (bool): enable self-attention blocks.
- `attn_heads` (int): attention heads.
- `attn_dim_head` (int): attention head dimension.
- `spatial_dims` (int): 1, 2, or 3.
- `emb_channels` (int or null): conditioning embedding width for residual blocks.
- `use_scale_shift_norm` (bool): enable scale-shift conditioning in residual blocks.
- `ckpt_path` (string or null): optional model init checkpoint.
- `norm_type` (string): factory-only residual-block norm override.
- `act` (string): factory-only residual-block activation override.

KL-only model keys:

- `double_z` (bool): encoder outputs mean/log-variance pairs.
- `norm_groups` (int or null): explicit GroupNorm groups.
- `codebook_size` / `num_embeddings` (int or null): accepted by `AutoencoderKL` for compatibility, but not used to instantiate a real codebook.

VQ-only model keys:

- `codebook_size` (int): number of embeddings.
- `vq_beta` (float): commitment cost.
- `vq_ema_decay` (float): EMA decay for `quantizer_type="ema"`.
- `vq_ema_eps` (float): numerical stabilizer for EMA updates.
- `quantizer_type` (string): `classic` or `ema`.
- `discriminator_type` (string): `patchgan` or `magvit`.

### Diffusion / flow model keys (`model.unet` and `model.scheduler`)

Top-level model keys:

- `model_type` (string): `diffusion` or `flow_matching`.

`model.unet` keys currently observed in configs:

- `in_channels`
- `out_channels`
- `block_out_channels`
- `layers_per_block`
- `attention_resolutions`
- `cross_attention_resolutions`
- `cross_attention_dim`
- `cross_attention_in_middle`
- `sample_size`
- `down_block_types`
- `up_block_types`

`model.scheduler` keys currently observed in configs:

- `name`
- `num_train_timesteps`
- `num_inference_steps`
- `params`

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

Observed `dataset.json` keys in this repo:

- `dataset_class` (required): import path in `module:Symbol` form
- `preprocess_kwargs` (object): forwarded into dataset constructors that accept it
- `data_root` (string): optional override from the dataset selector

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

Supported `run_model.py` modes:

- `sample`
- `encode`
- `decode`
- `evaluate`
- `build_tensor_cache`
- `debug_compare`

Core runtime flags:

- `--ckpt_dir`
- `--mode`
- `--data_txt`
- `--save`
- `--output_dir`
- `--batch_size`
- `--device`
- `--seed`
- `--timestep`
- `--num_samples`
- `--num_inference_steps`
- `--start_step`
- `--last_n_steps`
- `--scheduler` (`ddpm`, `ddim`, `dpmsolver1`, `dpmsolver2`, `dpmsolver++`, `dpmsolversde`, `unipc`, `flowmatch`)
- `--save_input`
- `--save_conditioning`
- `--save_tensor_cache`

Notes:

- `build_tensor_cache` writes cache files when `training.save_tensor_cache=true` or `--save_tensor_cache` is passed.
- Evaluation metrics are written to `--output_dir` when provided; otherwise to `--ckpt_dir`.
