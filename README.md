# LDCT_v2

LDCT_v2 is a lightweight research codebase for training latent autoencoders and diffusion-ready UNets on Low-Dose CT (LDCT) data. The repository mirrors the Stable Diffusion VAE defaults while exposing reusable neural building blocks that generalise to 1D/2D/3D workloads.

## Project Layout

- `src/` – Python package with reusable modules.
  - `nn/` – Core neural primitives (residual blocks, attention, pooling/upsampling, time embeddings).
  - `models/` – High-level model assemblies (AutoencoderKL, EfficientUNetND).
  - `pipelines/` – Train/eval entry points. Currently `train_vae.py` trains the VAE on CT slices.
  - `utils/` – Dataset loaders and helper utilities.
- `train_vae.py` – Thin wrapper so `python train_vae.py` still works after moving the pipeline under `src`.
- `checkpoints/` – Output directory for model checkpoints, configs, and per-epoch metric logs (ignored by git).
- `run_tests.py` – Placeholder pytest runner.

## Training the VAE

```
python -m src.pipelines.train_vae \
  --data-root /path/to/DefaultDataset \
  --output-dir checkpoints/vae_run \
  --epochs 100 \
  --batch-size 4 \
  --slice-count 1 \
  --img-size 256 \
  --use-amp
```

Key behaviours:
- The script logs per-epoch train/val metrics to `<output-dir>/metrics.jsonl` and stores the best checkpoint as `vae_best_epochXXXX.pt`.
- Inputs are normalised to `[-1, 1]`. Any spatial mismatch with the model resolution is handled automatically by interpolation (warned once per loop).
- Mixed precision can be toggled with `--use-amp` to reduce VRAM use.

See the README inside each subdirectory for detailed documentation of the provided modules.***
