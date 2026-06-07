# Distillation Workflow

Step-by-step process to train a teacher checkpoint, distill it into a faster student, and iterate in rounds.

## Overview

Distillation in this codebase uses `model_type: "distillation"` with:

- `model.teacher_checkpoint`: path to the teacher checkpoint (`*.pt`)
- `model.teacher_steps`: teacher step budget (for experiment bookkeeping)
- `model.student_steps`: student step budget (must be lower than teacher)
- `model.student_model_type`: student architecture family (`diffusion` by default)

The trainer freezes the teacher and supports two distinct training modes:

- `training.distillation_mode: "feature_matching"`
  - the student matches the teacher prediction at the same noisy input
  - `teacher_steps` / `student_steps` remain experiment-budget metadata only
- `training.distillation_mode: "progressive"`
  - the teacher is rolled out for multiple compressed substeps
  - the student is trained against an equivalent one-step target in the model family's native prediction space
  - `teacher_steps` / `student_steps` actively affect the training target

Current progressive-family scope:

- `diffusion`: DDPM epsilon-space target
- `flow_matching`: reverse-direction velocity target
- `rectified_flow`: forward-direction velocity target
- `edm`: sigma-aware noise-space target

This is still an engineering-first implementation, not a claim of theory-complete equivalence across all scheduler
families. In particular, the DDPM progressive target uses an approximation when projecting back into epsilon space.

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
- `training.distillation_mode: "feature_matching"` or `"progressive"`

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
    "distillation_mode": "feature_matching",
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
- Use `feature_matching` when you want a conservative teacher/student baseline.
- Use `progressive` when you want teacher/student step budgets to change the actual training target.
- Keep teacher/student architecture-compatible unless you intentionally handle mismatch externally.
