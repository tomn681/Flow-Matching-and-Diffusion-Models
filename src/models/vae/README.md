# `src.models.vae`

Implementation of modular VAEs:
- `AutoencoderKL`: KL-regularized latent autoencoder with the default PatchGAN discriminator returned by `make_discriminator()`
- `VQVAE`: shared VQ autoencoder whose quantizer and discriminator are selected from config

Paper-level VQ recipe choices remain in config:
- `quantizer_type`: `classic` or `ema`
- `discriminator_type`: `patchgan` or `magvit`

Defaults mirror the common 2D SD-style autoencoder shape (`resolution=256`, `base_ch=128`, `ch_mult=(1, 2, 4, 4)`, `embed_dim=4`). Configs can alternatively supply explicit `down_channels` (stage widths) and override the residual-block norm/activation through `norm_type` / `act`.

## Architecture Walkthrough

### Encoder

1. **Stem:** `ConvND(spatial_dims, in_channels, base_ch, kernel_size=3, padding=1)` produces the first 128-channel feature map in 2D mode.
2. **Down stages:** For each multiplier in `ch_mult` (or each entry in `down_channels`):
   - Run `num_res_blocks` residual blocks (default 2) using `ResBlockND`. Each block promotes the channel width to `base_ch * mult` (or the explicit stage width).
   - If the current spatial resolution (tracked internally) is listed in `attn_resolutions`, append `SpatialSelfAttention` after the block.
   - Add a `DownsampleND` (stride-2) layer unless this is the final stage. With the defaults the encoder halves the spatial size three times: 256 → 128 → 64 → 32.
3. **Bottleneck:** Two additional `ResBlockND` modules sandwich an optional attention block.
4. **Output head:** `GroupNorm → SiLU → ConvND` produces either `z_channels` channels or `2 * z_channels` when `double_z=True` (the default). The latter is split into mean/log-variance to form a `DiagonalGaussian` posterior. `norm_groups` (when provided) sets the GroupNorm groups; otherwise a sensible default is chosen.

### Latent Projections

- `quant_conv` (1×1) maps the encoder output to `2 * embed_dim` channels prior to sampling.
- After sampling (or taking the mode), `post_quant_conv` (1×1) projects the latent back to `z_channels` so the decoder can ingest it.
- `LATENT_SCALE = 0.18215` matches the Stable Diffusion convention. `AutoencoderKL.encode(..., normalize=True)` returns a scaled latent tensor ready for diffusion models.
- For VQ variants, `codebook_size` sets the number of embeddings; KL variants ignore the codebook even if present in the config.
- `quantizer_type="classic"` matches the original VQ-VAE codebook loss with direct codebook gradients.
- `quantizer_type="ema"` uses EMA codebook updates, closer to later EMA-VQ / VQGAN-style tokenizers.
- `discriminator_type="magvit"` selects the n-dimensional MAGVIT-style discriminator while preserving the same shared `VQVAE` architecture.
- `discriminator_type` only matters when the training recipe enables GAN loss (`gan_weight > 0`).

### Decoder

1. **Input:** `ConvND(spatial_dims, z_channels, base_ch * ch_mult[-1]` (or the last `down_channels` entry), `kernel_size=3, padding=1)` lifts the latent to the deepest feature width.
2. **Bottleneck:** Another `ResBlock → Attention → ResBlock` stack mirrors the encoder’s middle stage.
3. **Up stages:** Iterate through `ch_mult` in reverse:
   - Run `num_res_blocks + 1` residual blocks (default 3) at the current width. The extra block matches the original SD architecture and gives two refinement blocks even after channel changes.
   - Inject attention after each block when the running resolution matches `attn_resolutions`.
   - Apply `UpsampleND` (nearest-neighbour + conv) between stages until the original resolution is restored.
4. **Projection:** `GroupNorm → SiLU → ConvND` maps back to `out_channels` (1 for grayscale CT, 3 for RGB) followed by an optional `tanh` (disabled by default). `norm_groups` can be set to override the GroupNorm grouping.

## API Surface

- `AutoencoderKL.encode(x, normalize=False)` → posterior or latent grid.
- `AutoencoderKL.decode(z, denorm=False)` → reconstruction.
- `AutoencoderKL.forward(x, sample_posterior=True)` → tuple `(recon, posterior)`.
- `VQVAE.encode(x, normalize=False)` → pre-quantization latent grid.
- `VQVAE.decode(z, denorm=False)` → reconstruction from quantized latents.
- `VQVAE.forward(x)` → tuple `(recon, {"vq_loss", "perplexity", "codes"})`.

### Checkpointing

Passing `ckpt_path` loads weights into the model; otherwise a warning is emitted and the model starts from scratch.

## Usage Tips

- Switch `spatial_dims=3` to turn the network into a volumetric VAE. The dataset must then provide `[B, C, D, H, W]` tensors and the interpolation helpers in the training loop should be extended to trilinear mode (currently implemented for 2D and 3D automatically).
- Attention is optional but can be enabled per downsample level by populating `attn_resolutions` with the corresponding factors (e.g. `(8,)` for the 32×32 feature map when the input is 256×256).
- `embed_dim` controls the latent channel count; the spatial dimensions are defined by `(resolution / 2**(len(ch_mult)-1))` or by the length of `down_channels`. For the default settings this yields 32×32×4 latents.

## Training / Sampling

- Build from JSON: `from models import build_from_json; model = build_from_json("configs/autoencoder_kl.json")`.
- Train with `(dataset, json_path)`: `from pipelines.train.vae_lib import train; train(train_ds, "configs/autoencoder_kl.json", val_ds)`.
- Loss composition includes `recon_type` in `{l1,mse,bce,bce_focal}`, optional LPIPS, optional GAN, KL annealing, and step-based GAN warmup via `gan_start_steps`.
- KL vs VQ is selected by `latent_type` / `reg_type`; the codebook loss is ignored outside VQ.
- Sampling remains available via `pipelines/samplers/vae.py` (recon + random latents from checkpoints).
