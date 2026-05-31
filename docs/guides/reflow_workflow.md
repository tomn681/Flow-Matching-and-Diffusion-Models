# Rectified Flow with Reflow Workflow

Step-by-step process to train a base flow model, generate coupling pairs, and train reflow rounds.

## 1. Train Base Flow Model

```bash
python3 train.py --config configs/experiments/pixel_fm_concat_ldct.json
```

## 2. Generate Coupling Pairs

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/pixel_fm_concat/ \
  --mode generate_reflow_pairs \
  --num_pairs 50000 \
  --batch_size 32 \
  --output_dir checkpoints/pixel_fm_concat/reflow_pairs
```

## 3. Train Reflow Round 1

Use a reflow config pointing to the generated pairs directory, for example:

- `training.reflow_pairs_dir: "checkpoints/pixel_fm_concat/reflow_pairs"`
- `model.model_type: "reflow"`

```bash
python3 train.py --config configs/experiments/reflow_round1_concat_ldct.json
```

## 4. Evaluate Step Reduction

Evaluate progressively lower inference steps (for example 50, 25, 10, 5, 1):

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/reflow_r1/ \
  --mode evaluate \
  --num_inference_steps 10
```

## 5. Optional Reflow Round 2

Generate new pairs from round-1 model, then train round 2:

```bash
python3 run_model.py \
  --ckpt_dir checkpoints/reflow_r1/ \
  --mode generate_reflow_pairs \
  --num_pairs 50000 \
  --batch_size 32
```

```bash
python3 train.py --config configs/experiments/reflow_round2_concat_ldct.json
```

## Notes

- `generate_reflow_pairs` is supported by diffusion-like generative samplers.
- Keep output directories isolated per round to avoid mixing pairs/checkpoints.
- If you run with custom scheduler/steps, keep them consistent between pair generation and reflow training for controlled comparisons.

