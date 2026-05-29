# Getting Started

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python -m genlib train --config configs/LDCT/LDCT_autoencoder_kl_test.json
```

## Inference

```bash
python -m genlib sample --ckpt_dir <run_dir>
python -m genlib evaluate --ckpt_dir <run_dir>
```

## Compatibility Commands

```bash
python train.py --config <config>
python run_model.py --ckpt_dir <run_dir> --mode sample
```
