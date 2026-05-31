# LoRA Fine-Tuning

This project supports optional LoRA injection during training via `training.lora`.

## What Works Today

LoRA is injected into `nn.Linear` modules whose names match target suffixes.
Default targets are:

- `to_q`
- `to_k`
- `to_v`
- `to_out.0`

This is a good fit for attention-heavy models that expose these projection names.

## Important Limitation

LoRA is **not automatically universal** across every architecture.
If your model does not have matching linear module names, training fails fast with:

- `No target Linear modules matched for LoRA injection...`

In that case, set `training.lora.target_modules` to suffixes that exist in your model.

## Minimal Config Example

```json
{
  "training": {
    "lora": {
      "enabled": true,
      "rank": 4,
      "alpha": 1.0,
      "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]
    }
  }
}
```

## Train With LoRA

```bash
python3 train.py --config <config.json>
```

## Save / Load LoRA-Only Weights

Use the helper API:

```python
from training.lora import LoRAWrapper

LoRAWrapper.save_lora_weights(model, "lora_only.pt")
LoRAWrapper.load_lora_weights(model, "lora_only.pt")
```

The model must already be LoRA-wrapped before loading LoRA-only weights.

