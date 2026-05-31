# Training Workflows

## Config-Driven Dispatch

`train.py` (and `python -m genlib train`) dispatches by `model.model_type`.

Supported families include:

- `vae`
- `unet`
- `diffusion`
- `flow_matching`
- `latent_diffusion`
- `latent_flow_matching`
- `latent_rectified_flow`
- `consistency`
- `edm`
- `rectified_flow`
- `reflow`
- `gan`

## Resume

```bash
python -m genlib train --config <config.json> --resume <checkpoint.pt>
```

## Latent Cache Pre-encoding

```bash
python -m genlib train --mode encode_latents --config <config.json>
```

## LoRA Fine-Tuning

See:

- `docs/tutorials/lora.md`
