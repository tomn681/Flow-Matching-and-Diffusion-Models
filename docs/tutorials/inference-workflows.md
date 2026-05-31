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

## Reflow Pair Generation

```bash
python -m genlib sample \
  --mode generate_reflow_pairs \
  --ckpt_dir <run_dir> \
  --num_pairs 50000 \
  --batch_size 32 \
  --output_dir <run_dir>/reflow_pairs
```
