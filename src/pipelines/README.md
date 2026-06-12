# `src.pipelines`

Higher-level orchestration and compatibility entry points.

Boundary:
- canonical training lives in `src.training`
- canonical runtime samplers live in `src.sampling`
- `src.pipelines.train.*_lib` and `src.pipelines.samplers` remain compatibility /
  delegation layers where needed
- `src.pipelines.utils.py` is a backward-compatible re-export surface for
  scheduling/sampling helpers owned by `src.scheduling`

## Trainers

### `train/vae_lib.py`

- Compatibility wrapper only. Consumes `(dataset, json_path, val_dataset)` and
  now forwards to the registered VAE trainer after emitting a deprecation
  warning.
- Features: `recon_type` in `{l1,mse,bce,bce_focal}`, optional LPIPS/patch-GAN losses, KL vs VQ via `reg_type`, KL annealing, GAN warmup via `gan_start` or `gan_start_steps`, auto micro-batching on OOM (opt out with `allow_microbatching=false`), AMP (`use_amp`), configurable LR schedulers (`training.lr_scheduler`), and checkpointing (`vae_best.pt`, `vae_last.pt`, plus epoch snapshots).
- Example: `python train.py --config configs/autoencoder_kl.json`

### `train/flow_matching_lib.py`

- Compatibility wrapper only. Forwards to the registered flow-matching trainer
  after emitting a deprecation warning.
- Uses gradient accumulation, AMP, and distributed training (`torchrun --nproc_per_node=N ...`). Distributed runs shard data via `DistributedSampler`, reduce metrics, and write checkpoints only from rank 0.
- Example: `python train.py --config configs/flow_matching/ldct_flow_matching.json`

### `train/diffusion_lib.py`

- Compatibility wrapper only. Forwards to the registered diffusion trainer
  after emitting a deprecation warning.

## Sampling / Encoding / Decoding

- Dispatcher: `python run_model.py --ckpt_dir <run_dir> --mode {sample,encode,decode,evaluate,...}`.
- Samplers live under `pipelines/samplers/` and are invoked via handler classes under `pipelines/samplers/handlers/`.
- Diffusion and flow-matching sampling share a common engine in `pipelines/samplers/diffusion_like.py` (thin wrappers in `diffusion.py` and `flow_matching.py`).
- Extended modes: `build_tensor_cache`, `debug_compare`, `generate_reflow_pairs`.
- Runtime controls: `--num_inference_steps`, `--start_step`, `--last_n_steps`, `--scheduler`, `--num_samples`, `--save_input`, `--save_conditioning`, `--save_tensor_cache`, `--num_pairs`.
- Metrics behavior: in `evaluate` mode with `--output_dir`, the sampler creates a unique experiment subfolder and writes metrics/per-image files there; otherwise it writes to `--ckpt_dir`.

Programmatic usage:

```python
from sampling import VAESampler

sampler = VAESampler(ckpt_dir="checkpoints/ldct_vae_test_run1", save=True)
sampler.sample()
sampler.encode()
sampler.decode()
sampler.evaluate()
```

## Validation / Multi-GPU

`train.py` automatically builds train/validation datasets from the JSON `training` section (uses `test.txt` for validation). Flow/diffusion trainers honour distributed settings when launched via `torchrun`, while the VAE trainer remains single-process but supports custom device overrides and micro-batching.
