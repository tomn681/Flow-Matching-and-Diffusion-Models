# Distillation Workflow

Step-by-step process to train a teacher checkpoint, distill it into a faster student, and iterate in rounds.

## Overview

Distillation in this codebase uses `model_type: "distillation"` with:

- `model.teacher_checkpoint`: path to the teacher checkpoint (`*.pt`)
- `model.teacher_steps`: teacher step budget (for experiment bookkeeping)
- `model.student_steps`: student step budget (must be lower than teacher)
- `model.student_model_type`: student architecture family (`diffusion` by default)

The trainer freezes the teacher and optimizes the student to match teacher predictions on noisy inputs.

## 1. Train Teacher

```bash
python3 train.py --config configs/teacher_ddpm.json
```

Example result:

- `checkpoints/teacher_ddpm_run1/diff_best.pt`

## 2. Create Distillation Config (Round 1)

Set:

- `model.model_type: "distillation"`
- `model.student_model_type: "diffusion"`
- `model.teacher_checkpoint: "checkpoints/teacher_ddpm_run1/diff_best.pt"`
- `model.teacher_steps: 1000`
- `model.student_steps: 500`

Then train:

```bash
python3 train.py --config configs/distill_1000_to_500.json
```

## 3. Chain Rounds (Progressive Halving)

Use the previous student as the next teacher:

```bash
python3 train.py --config configs/distill_500_to_250.json
python3 train.py --config configs/distill_250_to_125.json
```

For each round:

- update `model.teacher_checkpoint` to the previous round best checkpoint
- set new `teacher_steps` / `student_steps`

## 4. Evaluate Distilled Checkpoints

Use normal sampling/eval flow on the distilled checkpoint directory:

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/distill_500_run1/ \
  --mode evaluate \
  --num_inference_steps 500
```

Repeat with multiple step counts for quality-speed comparison.

## Minimal Config Skeleton

```json
{
  "model": {
    "model_type": "distillation",
    "student_model_type": "diffusion",
    "teacher_checkpoint": "checkpoints/teacher_ddpm_run1/diff_best.pt",
    "teacher_steps": 1000,
    "student_steps": 500,
    "scheduler": { "name": "ddpm" },
    "unet": {
      "in_channels": 1,
      "out_channels": 1,
      "spatial_dims": 2
    }
  },
  "training": {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "output_dir": "checkpoints/distill_1000_to_500_run1"
  }
}
```

## Notes

- Distillation currently targets diffusion-family denoisers.
- `student_steps` must be strictly lower than `teacher_steps`.
- Keep teacher/student architecture-compatible unless you intentionally handle mismatch externally.
