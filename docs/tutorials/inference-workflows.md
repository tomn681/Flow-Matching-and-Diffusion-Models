# Inference Workflows

`python -m genlib` routes runtime modes to the sampler dispatcher.

## Modes

- `sample`
- `encode`
- `decode`
- `evaluate`
- `build_tensor_cache`
- `debug_compare`

## Example

```bash
python -m genlib sample --ckpt_dir <run_dir> --save --batch_size 4
```

## Scheduler Override

```bash
python -m genlib sample --ckpt_dir <run_dir> --scheduler ddim --num_inference_steps 50
```
