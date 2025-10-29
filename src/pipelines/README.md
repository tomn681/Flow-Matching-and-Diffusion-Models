# `src.pipelines`

Executable training and evaluation entry points. Currently the focus is on VAE training for LDCT slices.

## `train_vae.py`

Command-line training script that wraps `AutoencoderKL` and `DefaultDataset`.

### Key Features

- **Configuration defaults** align with Stable Diffusion’s VAE (`base_ch=128`, `ch_mult=(1, 2, 4, 4)`, `embed_dim=4`).
- **Dataset loader**: `DefaultDataset` returns dictionaries with `target` tensors; optional `image` fields (LDCT) are ignored when training with `diff=True`.
- **Data normalisation**: Inputs are scaled from `[0, 1]` into `[-1, 1]`.
- **Automatic resizing**: If dataset slices do not match the VAE resolution, they are interpolated (with warnings) so that reconstruction loss is well-defined.
- **Mixed precision**: Enable with `--use-amp` to activate autocast + GradScaler.
- **Checkpointing**: Periodic saves every `--save-every` epochs plus a running best checkpoint (`vae_best_epochXXXX.pt`).
- **Metric logging**: Per-epoch JSON Lines file (`metrics.jsonl`) records train/val losses and learning rate for downstream plotting.

### CLI Summary

```
python -m src.pipelines.train_vae \
  --data-root <dataset> \
  --output-dir checkpoints/vae_run \
  [--epochs 100] [--batch-size 4] [--num-workers 4]
  [--img-size 256] [--channel-mult 1,2,4,4] [--attn-resolutions 16]
  [--use-amp] [--resume path/to/checkpoint.pt]
```

Short aliases (`-d`, `-o`, `-e`, `-b`, …) are provided for interactive usage. See `python -m src.pipelines.train_vae --help` for the full list.***
