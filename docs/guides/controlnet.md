# ControlNet Workflow

This guide covers training and runtime usage for `ControlNetND`.

## 1. What ControlNet Adds

ControlNet keeps a base UNet backbone and adds zero-initialized control
residuals driven by a conditioning input.

The important property is that zero initialization preserves the original model
behavior before control branches learn anything useful.

## 2. Training Status

`ControlNetTrainer` is now registered under `model.model_type: "controlnet"`.

Current scope:

- DDPM epsilon-prediction training only
- frozen base UNet loaded from `model.base_unet_checkpoint`
- trainable `ControlNetND` initialized from the base UNet encoder/mid weights

A minimal config needs:

- `model.model_type: "controlnet"`
- `model.base_unet_checkpoint`
- a `model.controlnet` block or compatible top-level ControlNet fields
- paired target / conditioning data where:
  - `target` is the supervised clean image
  - `image` is the control input

Train with:

```bash
python3 train.py --config configs/<controlnet_config>.json
```

## 3. Sample with the Trained Checkpoint

`ControlNetSampler` is registered under `model.model_type: "controlnet"`, so
the standard runtime entrypoint now works for ControlNet checkpoints:

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/<controlnet_run> \
  --data_txt data/<split>.txt \
  --mode sample \
  --save
```

Evaluation uses the same path:

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/<controlnet_run> \
  --data_txt data/<split>.txt \
  --mode evaluate \
  --save
```

## 4. Conditioning Requirements

The dataset must provide:

- `target`
- conditioning input under the expected key for the configured adapter

For image-like control inputs, the most common pattern is channel concatenation
or explicit attention conditioning depending on the base model.

## Notes

- The codebase already includes zero-init residual regression coverage.
- The first trainer version is DDPM-only by design. It does not pretend to be
  scheduler-family generic yet.
- Keep conditioning resolution aligned with the target unless the adapter
  explicitly resizes it.
