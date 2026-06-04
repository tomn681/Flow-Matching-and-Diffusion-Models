# LoRA Fine-Tuning Workflow

LoRA support is integrated at training time through `training.lora`.

## 1. Enable LoRA in Config

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

## 2. Train

```bash
python3 train.py --config configs/<config>.json
```

`BaseTrainer` applies the wrapper before optimizer construction, so only LoRA
parameters remain trainable when the targets match.

## 3. Save and Load LoRA-Only Weights

```python
from training.lora import LoRAWrapper

LoRAWrapper.save_lora_weights(model, "lora_only.pt")
LoRAWrapper.load_lora_weights(model, "lora_only.pt")
```

## 4. Important Limitation

LoRA injection is not automatic for every architecture. The wrapper searches
for `nn.Linear` modules whose names end with the configured suffixes.

If nothing matches, the framework fails fast instead of pretending LoRA was
applied.

## 5. Recommended Workflow

1. verify the target module names on the chosen architecture
2. run a one-step smoke test
3. train with LoRA enabled
4. save LoRA-only weights separately if you want compact adapters
