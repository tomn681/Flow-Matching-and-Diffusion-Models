# ControlNet Workflow

This guide covers training and runtime usage for `ControlNetND`.

## 1. What ControlNet Adds

ControlNet keeps a base UNet backbone and adds zero-initialized control
residuals driven by a conditioning input.

The important property is that zero initialization preserves the original model
behavior before control branches learn anything useful.

## 2. Train a ControlNet Model

Use a config with:

- `model.model_type: "controlnet"`
- a compatible conditioning mode
- paired target / conditioning data

Example invocation:

```bash
python3 train.py --config configs/<controlnet_config>.json
```

## 3. Sample with the Trained Checkpoint

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/<controlnet_run_dir> \
  --mode sample
```

## 4. Conditioning Requirements

The dataset must provide:

- `target`
- conditioning input under the expected key for the configured adapter

For image-like control inputs, the most common pattern is channel concatenation
or explicit attention conditioning depending on the base model.

## Notes

- The codebase already includes zero-init residual regression coverage.
- Keep conditioning resolution aligned with the target unless the adapter
  explicitly resizes it.
