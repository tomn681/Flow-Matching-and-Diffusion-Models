# `src.pipelines`

Executable training and evaluation entry points. Training is organized under `src/pipelines/train/` and dispatched via `python train.py --config <json>` (CLI) or `from pipelines.train import train_*` (library). `src/pipelines/utils.py` contains shared helpers for schedulers/conditioning.

## Trainers

### `train/vae_lib.py`

- Consumes `(dataset, json_path, val_dataset)` and builds the VAE via `utils.model_utils.vae_utils.build_vae_model`.
- Features: `recon_type` in `{l1,mse,bce,bce_focal}`, optional LPIPS/patch-GAN losses, KL vs VQ via `reg_type`, KL annealing, auto micro-batching on OOM (opt out with `allow_microbatching=false`), AMP (`use_amp`), configurable schedulers (`training.scheduler`), and checkpointing (`vae_best.pt`, `vae_last.pt`, plus epoch snapshots/samples).
- Example: `python train.py --config configs/vae.json`

### `train/flow_matching_lib.py`

- Loads Diffusers UNet2D models via `utils.model_utils.diffusion_utils.build_diffusion_model`, drives FlowMatchEuler schedulers, and supports LDCT conditioning (`conditioning: "concatenate"`).
- Uses cosine warmup (`lr_warmup_steps`), gradient accumulation, AMP, and distributed training (`torchrun --nproc_per_node=N ...`). Distributed runs shard data via `DistributedSampler`, reduce metrics, and write checkpoints/samples only from rank 0.
- Example: `python train.py --config configs/flow_matching/ldct_flow_matching.json`

### `train/diffusion_lib.py`

- Implements DDPM-style training with schedulers named in the config (`ddpm`, `ddim`, `dpm_multistep`, etc.). Shares conditioning/AMP/gradient features with the flow-matching trainer.
- Saves `diff_last.pt`, `diff_best.pt`, per-epoch checkpoints, and visual grids under `<output_dir>/visuals`.

## Sampling / Encoding / Decoding

- Dispatcher: `python run_model.py --ckpt_dir <run_dir> --mode {sample,encode,decode,evaluate}`.
- Samplers live under `pipelines/samplers/` and are invoked via handler classes under `pipelines/samplers/handlers/`.

## Validation / Multi-GPU

`train.py` automatically builds train/validation datasets from the JSON `training` section (uses `test.txt` for validation). Flow/diffusion trainers honour distributed settings when launched via `torchrun`, while the VAE trainer remains single-process but supports custom device overrides and micro-batching.
