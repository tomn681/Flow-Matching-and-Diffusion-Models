# Getting Started

This is the shortest end-to-end path through the library: install, train a
small VAE, and sample from the resulting checkpoint.

## 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Train a Small VAE

```bash
python3 train.py --config configs/LDCT/LDCT_autoencoder_kl_test.json
```

Equivalent unified CLI:

```bash
python -m genlib train --config configs/LDCT/LDCT_autoencoder_kl_test.json
```

## 3. Sample or Reconstruct

```bash
python3 run_model.py --ckpt_dir checkpoints/<run_dir> --mode sample
python3 run_model.py --ckpt_dir checkpoints/<run_dir> --mode evaluate
```

Equivalent unified CLI:

```bash
python -m genlib sample --ckpt_dir checkpoints/<run_dir>
python -m genlib evaluate --ckpt_dir checkpoints/<run_dir>
```

## 4. Resume Training

```bash
python3 train.py --config configs/LDCT/LDCT_autoencoder_kl_test.json --resume checkpoints/<run_dir>/vae_last.pt
```

## Notes

- `train.py` dispatches by `model.model_type`.
- `run_model.py` and `python -m genlib sample` dispatch by checkpoint config.
- The saved run directory always contains `train_config.json`, so runtime modes can
  reconstruct the original model/dataset settings.
