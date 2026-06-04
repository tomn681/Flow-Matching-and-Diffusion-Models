# Latent Diffusion Workflow

Latent diffusion is a two-stage process:

1. train a VAE
2. train a latent-space generative model on encoded latents

## 1. Train the VAE

```bash
python3 train.py --config configs/LDCT/vae/vae_sd_kl_bce_focal_ldct.json
```

Keep the resulting checkpoint directory. The latent trainer needs the saved
`train_config.json` plus the VAE weights.

## 2. Optional: Pre-encode Latents

If your config enables latent caching or pre-saved latents, build them first:

```bash
python -m genlib train --mode encode_latents --config configs/<latent_config>.json
```

This reduces repeated VAE encoding cost during latent training.

## 3. Train a Latent Model

Latent model families currently include:

- `latent_diffusion`
- `latent_flow_matching`
- `latent_rectified_flow`

Example:

```bash
python3 train.py --config configs/<latent_diffusion_config>.json
```

## 4. Sample Through the Latent Sampler

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/<latent_run_dir> \
  --mode sample
```

The runtime:

- loads the latent model
- loads the referenced VAE
- predicts latent samples
- decodes them back to image space

## Notes

- `training.recon_type` affects latent decode visualization because the runtime
  uses the same reconstruction conversion helper.
- If you use multi-resolution training, do not combine it with fixed-resolution
  pre-saved latents.
