# Inference Workflows

`python -m genlib` routes runtime modes to the sampler dispatcher.

## Modes

- `sample`
- `encode`
- `decode`
- `evaluate`
- `build_tensor_cache`
- `debug_compare`
- `generate_reflow_pairs`

## Example

```bash
python -m genlib sample --ckpt_dir <run_dir> --save --batch_size 4
```

## Scheduler Override

```bash
python -m genlib sample --ckpt_dir <run_dir> --scheduler ddim --num_inference_steps 50
```

## Runtime Image Source

`--conditioning auto` follows the dataset configuration. Use
`--conditioning image` for LDCT -> SDCT evaluation or `--conditioning target`
for SDCT -> SDCT. This selector works for VAE, diffusion, and flow-matching
samplers; it changes the source tensor, not the model's configured conditioning
architecture. Evaluation metrics always compare the output with the SDCT target.

## Reflow Pair Generation

```bash
python -m genlib sample \
  --mode generate_reflow_pairs \
  --ckpt_dir <run_dir> \
  --num_pairs 50000 \
  --batch_size 32 \
  --output_dir <run_dir>/reflow_pairs
```
