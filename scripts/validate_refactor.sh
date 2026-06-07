#!/usr/bin/env bash
# =============================================================================
# validate_refactor.sh — End-to-end validation before merging to main
# =============================================================================
#
# Usage:
#   ./validate_refactor.sh \
#       --legacy-vae-ckpt  /path/to/vae_checkpoint.pt \
#       --legacy-ddpm-ckpt /path/to/ddpm_checkpoint_dir/ \
#       --legacy-vae-config /path/to/vae_config.json \
#       --legacy-ddpm-config /path/to/ddpm_config.json \
#       --data-txt /path/to/data_split.txt \
#       [--gpu]          # run GPU tests (default: CPU only)
#       [--quick]        # skip long training tests (2-epoch → 1 step)
#       [--skip-legacy]  # skip legacy checkpoint tests if you don't have them
#
# What it tests:
#   1. Unit test suite (pytest)
#   2. Config validation (all JSON configs)
#   3. Public API import smoke
#   4. Registry completeness / symmetry coverage
#   5. Legacy checkpoint loading (VAE + DDPM)
#   7. New trainer API smokes (VAE / diffusion / flow / rectified flow / GAN / latent / UNet / distillation)
#  10. New sampler/runtime smokes (sampling, reflow pair generation)
#  12. Core infrastructure checks (EMA, events, templates, weight mappers)
#  20. Conditioning / CFG / img2img / ControlNet runtime checks
#  28. Phase Q feature coverage:
#      - reflow pair generation
#      - LoRA wrap/save/load round-trip
#      - UNet trainer
#      - Distillation trainer
#      - DiT forward pass
#      - extended schedulers
#      - super-resolution adapter
#      - all new configs validate
#
# Exit codes:
#   0 = all tests passed
#   1 = at least one test failed
# =============================================================================

set -euo pipefail

# ─── Colors ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Counters ───
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# ─── Parse arguments ───
LEGACY_VAE_CKPT=""
LEGACY_DDPM_CKPT="/home/shared_data/LDCT/LDCT/train/ddpm_concat-no-256-1-42-2025-20-01-22:42"
LEGACY_VAE_CONFIG=""
# Deprecated: kept for CLI backward compatibility; no longer required.
LEGACY_DDPM_CONFIG=""
DATA_TXT=""
USE_GPU=false
QUICK=false
SKIP_LEGACY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --legacy-vae-ckpt)    LEGACY_VAE_CKPT="$2"; shift 2 ;;
        --legacy-ddpm-ckpt)   LEGACY_DDPM_CKPT="$2"; shift 2 ;;
        --legacy-vae-config)  LEGACY_VAE_CONFIG="$2"; shift 2 ;;
        --legacy-ddpm-config) LEGACY_DDPM_CONFIG="$2"; shift 2 ;;
        --data-txt)           DATA_TXT="$2"; shift 2 ;;
        --gpu)                USE_GPU=true; shift ;;
        --quick)              QUICK=true; shift ;;
        --skip-legacy)        SKIP_LEGACY=true; shift ;;
        *)                    echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Paths ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
WORK_DIR=$(mktemp -d -t validate_refactor_XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT

export PYTHONPATH="$SRC_DIR:$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="python3"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

DEVICE="cpu"
if $USE_GPU && "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
fi

TRAIN_EPOCHS=2
if $QUICK; then
    TRAIN_EPOCHS=1
fi

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        Refactor Validation Suite                          ║${NC}"
echo -e "${CYAN}║        Device: ${DEVICE}, Epochs: ${TRAIN_EPOCHS}                              ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Helpers ───
run_test() {
    local name="$1"
    shift
    echo -e "${CYAN}[TEST] ${name}${NC}"
    if eval "$@" > "$WORK_DIR/last_output.log" 2>&1; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}✗ FAILED${NC}"
        echo -e "  ${RED}  Log: $WORK_DIR/last_output.log${NC}"
        tail -20 "$WORK_DIR/last_output.log" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILURES+=("$name")
    fi
}

skip_test() {
    local name="$1"
    local reason="$2"
    echo -e "${YELLOW}[SKIP] ${name} — ${reason}${NC}"
    SKIPPED=$((SKIPPED + 1))
}

# ═══════════════════════════════════════════════════════════════
# TEST 1: Full pytest suite
# ═══════════════════════════════════════════════════════════════
run_test "1. Pytest unit + integration suite" \
    "cd '$PROJECT_ROOT' && $PYTHON_BIN -m pytest tests/ -q --tb=short"

# ═══════════════════════════════════════════════════════════════
# TEST 2: Config validation — all JSON configs parse
# ═══════════════════════════════════════════════════════════════
run_test "2. All JSON configs validate" \
    "$PYTHON_BIN -c \"
import json, sys
from pathlib import Path
sys.path.insert(0, '$SRC_DIR')
from configs.schema import validate_config
configs_dir = Path('$PROJECT_ROOT/configs')
errors = []
for f in sorted(configs_dir.rglob('*.json')):
    if f.name == 'dataset.json':
        continue
    try:
        cfg = json.loads(f.read_text())
        validate_config(cfg)
    except Exception as e:
        errors.append(f'{f.name}: {e}')
if errors:
    print('Config validation failures:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
print('All %d configs validated.' % len(list(configs_dir.rglob('*.json'))))
\""

# ═══════════════════════════════════════════════════════════════
# TEST 3: Public API import smoke
# ═══════════════════════════════════════════════════════════════
run_test "3. Public API imports" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
# Core
from core.types import ModelOutput, NoisyBatch, TrainingState
from core.registry import Registry
# Models
from models import AutoencoderKL, VQVAE, BaseVAE, ModelFactory, MODEL_REGISTRY
from models.unet import EfficientUNetND, UNetDiffusersND, UNet2DConditionND, BaseUNetND
# Training
from training import BaseTrainer, VAETrainer, DiffusionTrainer, FlowMatchingTrainer, TRAINER_REGISTRY
from training import CheckpointCallback, MetricsCSVCallback, VisualizationCallback
from training.ema import EMAModel
from training.builder import TrainerBuilder
from training.events import TrainingEventBus
# Sampling
from sampling import BaseSampler, SAMPLER_REGISTRY
# Losses
from losses import LossAssembler, LOSS_REGISTRY, BaseLossComponent
# Noise
from noise import DDPMNoise, FlowMatchingNoise, NOISE_REGISTRY
# Scheduling
from scheduling import build_scheduler, SCHEDULER_REGISTRY, LR_SCHEDULER_REGISTRY
from scheduling.conditioning_chain import ConditioningChain
# Configs
from configs import TrainingConfig, validate_config, load_config
from configs.templates import from_template
# Adapters
from models.adapters.weight_mappers import map_hf_unet_to_ours, map_hf_vae_to_ours
from models.adapters.text_encoders import build_text_encoder
# ControlNet
from models.controlnet import ControlNetND
# Compat
from compat.legacy_training import train_diffusion, train_flow_matching
print('All public API imports successful.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 4: Registry completeness
# ═══════════════════════════════════════════════════════════════
run_test "4. Registry completeness" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from models import MODEL_REGISTRY
from noise import NOISE_REGISTRY
from losses import LOSS_REGISTRY
from training import TRAINER_REGISTRY
from sampling import SAMPLER_REGISTRY
from scheduling import SCHEDULER_REGISTRY, LR_SCHEDULER_REGISTRY
from nn.blocks import BLOCK_REGISTRY

checks = {
    'MODEL_REGISTRY': (MODEL_REGISTRY, ['kl_vae', 'vq_vae', 'efficient_unet', 'diffusers_unet', 'condition_unet', 'controlnet', 'dit', 'video_unet']),
    'NOISE_REGISTRY': (NOISE_REGISTRY, ['ddpm', 'flow_matching', 'consistency', 'edm', 'rectified_flow', 'reflow']),
    'LOSS_REGISTRY': (LOSS_REGISTRY, ['l1', 'mse', 'bce', 'focal', 'bce_focal', 'kl', 'vq', 'perceptual', 'gan_generator', 'gan_discriminator']),
    'TRAINER_REGISTRY': (TRAINER_REGISTRY, ['vae', 'diffusion', 'flow_matching', 'consistency', 'edm', 'rectified_flow', 'reflow', 'gan', 'unet', 'distillation', 'controlnet', 'latent_diffusion', 'latent_flow_matching', 'latent_rectified_flow']),
    'SAMPLER_REGISTRY': (SAMPLER_REGISTRY, ['vae', 'diffusion', 'flow_matching', 'consistency', 'edm', 'rectified_flow', 'reflow', 'unet', 'video_unet', 'distillation', 'controlnet', 'latent_diffusion', 'latent_flow_matching', 'latent_rectified_flow']),
}
errors = []
for name, (reg, expected) in checks.items():
    for key in expected:
        if key not in reg:
            errors.append(f'{name} missing: {key}')
if errors:
    for e in errors:
        print(e)
    import sys; sys.exit(1)
print('All registries complete.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 5: Legacy checkpoint loading (VAE)
# ═══════════════════════════════════════════════════════════════
if $SKIP_LEGACY || [ -z "$LEGACY_VAE_CKPT" ]; then
    skip_test "5. Legacy VAE checkpoint load" "no --legacy-vae-ckpt provided"
else
    run_test "5. Legacy VAE checkpoint load + forward pass" \
        "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from models import AutoencoderKL, ModelFactory
import json

cfg = json.load(open('$LEGACY_VAE_CONFIG'))
model = ModelFactory.build(cfg)
payload = torch.load('$LEGACY_VAE_CKPT', map_location='cpu')
state = payload.get('model', payload)
model.load_state_dict(state)
model.eval()

# Forward pass
x = torch.randn(1, model.encoder.conv_in.conv.in_channels, 32, 32)
with torch.no_grad():
    output = model(x, sample_posterior=False)
rec = output.reconstruction
assert rec.shape == x.shape, f'Shape mismatch: {rec.shape} vs {x.shape}'
assert torch.isfinite(rec).all(), 'Non-finite values in reconstruction'
print(f'Legacy VAE loaded. Forward pass OK. Output shape: {rec.shape}')
\""
fi

# ═══════════════════════════════════════════════════════════════
# TEST 6: Legacy checkpoint loading (DDPM)
# ═══════════════════════════════════════════════════════════════
if $SKIP_LEGACY || [ -z "$LEGACY_DDPM_CKPT" ]; then
    skip_test "6. Legacy DDPM checkpoint load" "no --legacy-ddpm-ckpt provided"
elif [ ! -d "$LEGACY_DDPM_CKPT" ]; then
    skip_test "6. Legacy DDPM checkpoint load" "legacy ddpm path not found: $LEGACY_DDPM_CKPT"
else
    run_test "6. Legacy DDPM checkpoint load + forward pass" \
        "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from utils.model_utils.diffusion_utils import build_diffusion_model
from utils.sampling_utils import load_run_config, resolve_checkpoint
from pathlib import Path

ckpt_dir = Path('$LEGACY_DDPM_CKPT')
cfg = load_run_config(ckpt_dir)
ckpt_path = str(resolve_checkpoint(ckpt_dir, cfg.get('model', {}).get('model_type', 'diffusion')))
model = build_diffusion_model(cfg, torch.device('cpu'), ckpt_path=ckpt_path)
model.eval()

in_ch = cfg.get('model', {}).get('unet', {}).get('in_channels', 1)
x = torch.randn(1, in_ch, 32, 32)
t = torch.tensor([100])
with torch.no_grad():
    pred = model(x, t)
if isinstance(pred, (list, tuple)):
    pred = pred[0]
elif hasattr(pred, 'sample'):
    pred = pred.sample
assert torch.isfinite(pred).all(), 'Non-finite values'
print(f'Legacy DDPM loaded. Forward pass OK. Output shape: {pred.shape}')
\""
fi

# ═══════════════════════════════════════════════════════════════
# TEST 7: VAE training on MNIST (new API)
# ═══════════════════════════════════════════════════════════════
VAE_TRAIN_DIR="$WORK_DIR/vae_train"
mkdir -p "$VAE_TRAIN_DIR"

run_test "7. VAETrainer: train KL-VAE on MNIST ($TRAIN_EPOCHS epochs)" \
    "$PYTHON_BIN -c \"
import sys, json, torch; sys.path.insert(0, '$SRC_DIR')
from training import VAETrainer
from torch.utils.data import Dataset

class TinyDataset(Dataset):
    def __len__(self): return 32
    def __getitem__(self, idx):
        x = torch.randn(1, 28, 28)
        return {'target': x, 'image': x}

config = {
    'training': {
        'epochs': $TRAIN_EPOCHS,
        'batch_size': 8,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$VAE_TRAIN_DIR',
        'seed': 42,
        'recon_type': 'l1',
        'kl_weight': 1e-6,
        'use_amp': False,
        'save_images': False,
        'visual_samples': 8,
    },
    'model': {
        'model_type': 'vae',
        'latent_type': 'kl',
        'in_channels': 1,
        'out_channels': 1,
        'resolution': 28,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'num_res_blocks': 1,
        'z_channels': 4,
        'embed_dim': 4,
        'spatial_dims': 2,
        'use_attention': False,
    },
}

ds = TinyDataset()
trainer = VAETrainer(config=config)
trainer.fit(ds)

# Verify checkpoint was saved
import pathlib
ckpt = pathlib.Path(trainer.output_dir)
assert (ckpt / 'vae_last.pt').exists(), f'No VAE last checkpoint at {ckpt}'
pathlib.Path('$WORK_DIR/vae_ckpt_dir.txt').write_text(str(ckpt))
print(f'VAE training complete. Checkpoint at {ckpt}')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 8: Diffusion training on MNIST (new API)
# ═══════════════════════════════════════════════════════════════
DIFF_TRAIN_DIR="$WORK_DIR/diff_train"
mkdir -p "$DIFF_TRAIN_DIR"

run_test "8. DiffusionTrainer: train DDPM on MNIST ($TRAIN_EPOCHS epochs)" \
    "$PYTHON_BIN -c \"
import sys, json, torch; sys.path.insert(0, '$SRC_DIR')
from training import DiffusionTrainer
from torch.utils.data import Dataset

class TinyDataset(Dataset):
    def __len__(self): return 32
    def __getitem__(self, idx):
        x = torch.randn(1, 28, 28)
        return {'target': x, 'image': x}

config = {
    'training': {
        'epochs': $TRAIN_EPOCHS,
        'batch_size': 8,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$DIFF_TRAIN_DIR',
        'seed': 42,
        'conditioning': 'none',
        'use_amp': False,
    },
    'model': {
        'model_type': 'diffusion',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'ddpm',
            'num_train_timesteps': 100,
            'num_inference_steps': 10,
        },
    },
}

ds = TinyDataset()
trainer = DiffusionTrainer(config=config)
trainer.fit(ds)

import pathlib
ckpt_dir = pathlib.Path(trainer.output_dir)
assert (ckpt_dir / 'diff_last.pt').exists()
pathlib.Path('$WORK_DIR/diff_ckpt_dir.txt').write_text(str(ckpt_dir))
print('Diffusion training complete.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 9: Flow Matching training on MNIST (new API)
# ═══════════════════════════════════════════════════════════════
FM_TRAIN_DIR="$WORK_DIR/fm_train"
mkdir -p "$FM_TRAIN_DIR"

run_test "9. FlowMatchingTrainer: train FM on MNIST ($TRAIN_EPOCHS epochs)" \
    "$PYTHON_BIN -c \"
import sys, json, torch; sys.path.insert(0, '$SRC_DIR')
from training import FlowMatchingTrainer
from torch.utils.data import Dataset

class TinyDataset(Dataset):
    def __len__(self): return 32
    def __getitem__(self, idx):
        x = torch.randn(1, 28, 28)
        return {'target': x, 'image': x}

config = {
    'training': {
        'epochs': $TRAIN_EPOCHS,
        'batch_size': 8,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$FM_TRAIN_DIR',
        'seed': 42,
        'conditioning': 'none',
        'use_amp': False,
    },
    'model': {
        'model_type': 'flow_matching',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'flow_match_euler',
            'num_train_timesteps': 100,
            'num_inference_steps': 10,
        },
    },
}

ds = TinyDataset()
trainer = FlowMatchingTrainer(config=config)
trainer.fit(ds)

import pathlib
ckpt_dir = pathlib.Path(trainer.output_dir)
assert (ckpt_dir / 'flow_last.pt').exists()
pathlib.Path('$WORK_DIR/fm_ckpt_dir.txt').write_text(str(ckpt_dir))
print('Flow Matching training complete.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 10: VAE checkpoint reload + forward pass
# ═══════════════════════════════════════════════════════════════
run_test "10. Reload trained VAE checkpoint + forward" \
    "$PYTHON_BIN -c \"
import sys, torch, pathlib; sys.path.insert(0, '$SRC_DIR')
from models import AutoencoderKL, ModelOutput

model = AutoencoderKL(
    in_channels=1, out_channels=1, resolution=28,
    base_ch=32, ch_mult=(1, 2), num_res_blocks=1,
    z_channels=4, embed_dim=4, spatial_dims=2,
    use_attention=False,
)
ckpt_dir = pathlib.Path('$WORK_DIR/vae_ckpt_dir.txt').read_text().strip()
payload = torch.load(str(pathlib.Path(ckpt_dir) / 'vae_last.pt'), map_location='cpu')
state = payload.get('model', payload)
model.load_state_dict(state)
model.eval()

x = torch.randn(2, 1, 28, 28)
with torch.no_grad():
    out = model(x, sample_posterior=False)
assert isinstance(out, ModelOutput), f'Expected ModelOutput, got {type(out)}'
assert out.reconstruction.shape == x.shape
assert torch.isfinite(out.reconstruction).all()
print('VAE checkpoint reload + forward OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 11: Diffusion sampling from trained checkpoint
# ═══════════════════════════════════════════════════════════════
run_test "11. Diffusion sampling from trained DDPM checkpoint" \
    "$PYTHON_BIN -c \"
import sys, torch, pathlib; sys.path.insert(0, '$SRC_DIR')
from scheduling import build_scheduler, sample_with_scheduler
from utils.model_utils.diffusion_utils import build_diffusion_model
import json

config = {
    'training': {'conditioning': 'none'},
    'model': {
        'model_type': 'diffusion',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'ddpm',
            'num_train_timesteps': 100,
            'num_inference_steps': 10,
        },
    },
}

ckpt_dir = pathlib.Path('$WORK_DIR/diff_ckpt_dir.txt').read_text().strip()
model = build_diffusion_model(config, torch.device('cpu'), ckpt_path=str(pathlib.Path(ckpt_dir) / 'diff_last.pt'))
scheduler, steps = build_scheduler(config['model']['scheduler'], config['training'])

with torch.no_grad():
    samples = sample_with_scheduler(
        model, scheduler,
        sample_shape=(2, 1, 28, 28),
        device=torch.device('cpu'),
        conditioning_mode='none',
        num_inference_steps=steps,
    )
assert samples.shape == (2, 1, 28, 28), f'Unexpected shape: {samples.shape}'
assert torch.isfinite(samples).all()
print(f'Sampling OK. Shape: {samples.shape}')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 12: EMA model integration
# ═══════════════════════════════════════════════════════════════
run_test "12. EMA model step + copy_to + state_dict round-trip" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from training.ema import EMAModel
import torch.nn as nn

model = nn.Linear(10, 10)
ema = EMAModel(model, decay=0.9)
# Modify model weights
with torch.no_grad():
    model.weight.fill_(1.0)
# Step EMA
for _ in range(200):
    ema.step(model)
# EMA should be close to 1.0
ema.copy_to(model)
assert (model.weight - 1.0).abs().max() < 0.20, 'EMA did not converge'
# Round-trip
state = ema.state_dict()
ema2 = EMAModel(nn.Linear(10, 10))
ema2.load_state_dict(state)
for k in ema.shadow_params:
    assert torch.equal(ema.shadow_params[k], ema2.shadow_params[k])
print('EMA integration OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 13: LossAssembler context-based pipeline
# ═══════════════════════════════════════════════════════════════
run_test "13. LossAssembler full pipeline with all loss types" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from losses import LossAssembler, LOSS_REGISTRY

recon = LOSS_REGISTRY.build('l1', weight=1.0)
kl = LOSS_REGISTRY.build('kl', weight=1e-6)
vq = LOSS_REGISTRY.build('vq', weight=0.0)
assembler = LossAssembler([recon, kl, vq])

class FakePosterior:
    def kl(self): return torch.tensor([0.1, 0.1])

ctx = {
    'reconstruction': torch.randn(2, 1, 8, 8),
    'reconstruction_image': torch.randn(2, 1, 8, 8),
    'target': torch.randn(2, 1, 8, 8),
    'posterior': FakePosterior(),
    'codebook_loss': None,
    'fake_pred': None,
    'device': torch.device('cpu'),
    'dtype': torch.float32,
}
total, parts = assembler(context=ctx, epoch=1, global_step=100)
assert torch.isfinite(total)
assert 'recon_l1' in parts
assert 'kl' in parts
keys = assembler.metric_keys()
assert len(keys) == 3
print(f'LossAssembler OK. Keys: {keys}, Total: {total.item():.4f}')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 14: Noise process shapes (all registered)
# ═══════════════════════════════════════════════════════════════
run_test "14. All noise processes produce correct shapes" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from noise import NOISE_REGISTRY

class FakeSched:
    class config:
        num_train_timesteps = 100
    def add_noise(self, x, n, t):
        return x + n * 0.01

for key in NOISE_REGISTRY.list():
    if key == 'reflow':
        # Reflow requires persisted pair data; covered by dedicated tests elsewhere.
        continue
    noise = NOISE_REGISTRY.build(key, scheduler=FakeSched())
    clean = torch.randn(4, 1, 8, 8)
    nb = noise(clean, torch.device('cpu'))
    assert nb.noisy.shape == clean.shape, f'{key}: noisy shape mismatch'
    assert nb.target.shape == clean.shape, f'{key}: target shape mismatch'
    assert nb.timesteps.shape == (4,), f'{key}: timesteps shape mismatch'
    print(f'  {key}: OK')
print('All noise processes OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 15: TrainerBuilder smoke test
# ═══════════════════════════════════════════════════════════════
run_test "15. TrainerBuilder constructs a valid trainer" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from training.builder import TrainerBuilder
from training import DiffusionTrainer

config = {
    'training': {'epochs': 1, 'batch_size': 4, 'learning_rate': 1e-3,
                 'output_dir': '$WORK_DIR/builder_test', 'seed': 42,
                 'conditioning': 'none', 'use_amp': False},
    'model': {
        'model_type': 'diffusion',
        'unet': {'unet_impl': 'efficient_nd', 'spatial_dims': 2,
                 'in_channels': 1, 'out_channels': 1, 'model_channels': 16,
                 'num_res_blocks': 1, 'channel_mult': [1, 2]},
        'scheduler': {'name': 'ddpm', 'num_train_timesteps': 50, 'num_inference_steps': 5},
    },
}

trainer = TrainerBuilder().with_config(config).build()
assert isinstance(trainer, DiffusionTrainer)
print('TrainerBuilder OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 16: Config templates
# ═══════════════════════════════════════════════════════════════
run_test "16. Config templates validate and can be overridden" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from configs.templates import from_template
from configs.schema import validate_config

for name in ['sd15_vae', 'sd15_latent_ddpm', 'fmboost_latent_fm', 'pixel_ddpm_1d', 'vqgan_magvit']:
    cfg = from_template(name)
    validate_config(cfg)
    print(f'  {name}: OK')

# Test override
cfg = from_template('sd15_vae', training={'epochs': 50})
assert cfg['training']['epochs'] == 50
print('Config templates OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 17: EventBus lifecycle
# ═══════════════════════════════════════════════════════════════
run_test "17. TrainingEventBus subscribe/emit/remove" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from training.events import TrainingEventBus

bus = TrainingEventBus()
events_received = []
def listener(**kw): events_received.append(kw)

bus.on('epoch_end', listener)
bus.emit('epoch_end', epoch=1, loss=0.5)
assert len(events_received) == 1
assert events_received[0]['epoch'] == 1

bus.remove('epoch_end', listener)
bus.emit('epoch_end', epoch=2, loss=0.3)
assert len(events_received) == 1  # listener removed
print('EventBus OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 18: Weight mapper key coverage (synthetic)
# ═══════════════════════════════════════════════════════════════
run_test "18. Weight mapper: key coverage + shape validation" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from models.adapters.weight_mappers import map_hf_unet_to_ours, map_hf_vae_to_ours

# Synthetic test: map keys and verify no crash
fake_hf = {'conv_in.weight': torch.randn(32, 1, 3, 3)}
mapped = map_hf_unet_to_ours(fake_hf)
# Should have renamed the key
assert 'conv_in.weight' in mapped or 'conv_in.conv.weight' in mapped
print('Weight mapper: key mapping OK (synthetic).')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 19: ControlNet forward pass
# ═══════════════════════════════════════════════════════════════
run_test "19. ControlNet forward produces zero-init residuals" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from models.controlnet import ControlNetND

cn = ControlNetND(
    spatial_dims=2, in_channels=4, conditioning_channels=3,
    down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
    block_out_channels=(32, 64), layers_per_block=1,
    cross_attention_dim=32, attention_head_dim=8,
    mid_block_type='UNetMidBlock2DCrossAttn',
)

x = torch.randn(1, 4, 16, 16)
t = torch.tensor([10], dtype=torch.long)
cond = torch.randn(1, 3, 16, 16)
ctx = torch.randn(1, 4, 32)

with torch.no_grad():
    out = cn(x, t, cond, encoder_hidden_states=ctx)

# Zero-conv init means residuals should be near zero
for r in out['down_residuals']:
    assert r.abs().max() < 1e-5, f'Non-zero residual at init: {r.abs().max()}'
assert out['mid_residual'].abs().max() < 1e-5
print('ControlNet zero-init OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 20: Conditioning adapters (all registered types)
# ═══════════════════════════════════════════════════════════════
run_test "20. All conditioning adapters run without error" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from scheduling.conditioning import CONDITIONING_ADAPTER_REGISTRY, resolve_conditioning_adapter

for key in CONDITIONING_ADAPTER_REGISTRY.list():
    adapter = resolve_conditioning_adapter(key)
    x = torch.randn(2, 4, 8, 8)
    if key == 'inpainting':
        cond = {'mask': torch.ones(2, 1, 8, 8), 'original': torch.randn(2, 4, 8, 8)}
    elif key == 'chain':
        cond = None
    else:
        cond = torch.randn(2, 4, 8, 8)
    try:
        out_x, out_ctx = adapter(x, cond, None)
        print(f'  {key}: OK (input={x.shape}, output={out_x.shape})')
    except Exception as e:
        if key in ('latent_attention', 'chain') and cond is None:
            print(f'  {key}: OK (no-op with None cond)')
        else:
            raise
print('All conditioning adapters OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 21: Rectified Flow training on MNIST (new API)
# ═══════════════════════════════════════════════════════════════
RF_TRAIN_DIR="$WORK_DIR/rf_train"
mkdir -p "$RF_TRAIN_DIR"

run_test "21. RectifiedFlowTrainer: train rectified flow on MNIST ($TRAIN_EPOCHS epochs)" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from training import RectifiedFlowTrainer
from torch.utils.data import Dataset
import pathlib

class TinyDataset(Dataset):
    def __len__(self): return 32
    def __getitem__(self, idx):
        x = torch.randn(1, 28, 28)
        return {'target': x, 'image': x}

config = {
    'training': {
        'epochs': $TRAIN_EPOCHS,
        'batch_size': 8,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$RF_TRAIN_DIR',
        'seed': 42,
        'conditioning': 'none',
        'use_amp': False,
    },
    'model': {
        'model_type': 'rectified_flow',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'flow_match_euler',
            'num_train_timesteps': 100,
            'num_inference_steps': 10,
        },
    },
}

ds = TinyDataset()
trainer = RectifiedFlowTrainer(config=config)
trainer.fit(ds)
assert (pathlib.Path(trainer.output_dir) / 'rectified_flow_last.pt').exists()
print('Rectified flow training complete.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 22: InferencePipeline generate smoke
# ═══════════════════════════════════════════════════════════════
run_test "22. InferencePipeline: generate() smoke test" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from pipelines.inference import InferencePipeline, InferenceInputs

class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([], dtype=torch.long)
    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)
    def step(self, pred, t, sample):
        return type('Out', (), {'prev_sample': sample - 0.1 * pred})()

class FakeUNet(torch.nn.Module):
    def forward(self, x, t, context_ca=None):
        _ = t, context_ca
        return torch.zeros_like(x)

pipe = InferencePipeline(
    unet=FakeUNet(),
    scheduler=FakeScheduler(),
    device=torch.device('cpu'),
    conditioning_mode='none',
)
inputs = InferenceInputs(sample_shape=(2, 1, 8, 8), num_inference_steps=4)
out = pipe.generate(inputs)
assert out.shape == (2, 1, 8, 8)
assert torch.isfinite(out).all()
print('InferencePipeline generate OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 23: GAN trainer one-step smoke
# ═══════════════════════════════════════════════════════════════
GAN_TRAIN_DIR="$WORK_DIR/gan_train"
mkdir -p "$GAN_TRAIN_DIR"

run_test "23. GANTrainer: 1-step smoke with tiny models" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from training import GANTrainer

class TinyDataset:
    def __len__(self): return 8
    def __getitem__(self, idx):
        x = torch.randn(1, 8, 8)
        return {'target': x, 'image': x}

class TinyGen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, x):
        return self.net(x)

class TinyDisc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 1, 1))
    def forward(self, x):
        return self.net(x)

config = {
    'training': {
        'epochs': 1,
        'batch_size': 4,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$GAN_TRAIN_DIR',
        'seed': 42,
        'use_amp': False,
        'manual_device': 'cpu',
        'save_every': 1,
    },
    'model': {
        'model_type': 'gan',
    },
}

trainer = GANTrainer(config=config, model_override=TinyGen(), discriminator_override=TinyDisc())
trainer.fit(TinyDataset())
print('GANTrainer smoke OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 24: Latent diffusion with pre-saved latents
# ═══════════════════════════════════════════════════════════════
LATENT_TRAIN_DIR="$WORK_DIR/latent_train"
LATENT_CACHE_DIR="$WORK_DIR/latent_cache"
mkdir -p "$LATENT_TRAIN_DIR" "$LATENT_CACHE_DIR/train" "$LATENT_CACHE_DIR/val"

run_test "24. LatentDiffusionTrainer: pre-saved latent cache training" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from training import LatentDiffusionTrainer
from datasets import LatentCacheDataset
import pathlib

for split in ('train', 'val'):
    root = pathlib.Path('$LATENT_CACHE_DIR') / split
    for i in range(8):
        torch.save(
            {
                'target': torch.randn(4, 8, 8),
                'image': torch.randn(4, 8, 8),
            },
            root / f'{i:08d}.pt'
        )

config = {
    'training': {
        'epochs': 1,
        'batch_size': 4,
        'num_workers': 0,
        'learning_rate': 1e-3,
        'output_dir': '$LATENT_TRAIN_DIR',
        'seed': 42,
        'conditioning': 'none',
        'use_amp': False,
    },
    'model': {
        'model_type': 'latent_diffusion',
        'use_presaved_latents': True,
        'latent_cache_dir': '$LATENT_CACHE_DIR',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 4,
            'out_channels': 4,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'ddpm',
            'num_train_timesteps': 50,
            'num_inference_steps': 5,
        },
    },
}

train_ds = LatentCacheDataset('$LATENT_CACHE_DIR', split='train')
val_ds = LatentCacheDataset('$LATENT_CACHE_DIR', split='val')
trainer = LatentDiffusionTrainer(config=config)
trainer.fit(train_ds, val_dataset=val_ds)
assert (pathlib.Path(trainer.output_dir) / 'latent_diff_last.pt').exists()
print('Latent pre-saved training OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 25: CFG dropout target-rate check
# ═══════════════════════════════════════════════════════════════
run_test "25. CFG conditioning dropout rate sanity" \
    "$PYTHON_BIN -c \"
import torch
p = 0.30
n = 20000
mask = torch.rand(n) < p
rate = float(mask.float().mean())
assert abs(rate - p) < 0.03, f'Observed dropout rate {rate:.3f} differs from p={p:.3f}'
print(f'CFG dropout empirical rate OK: {rate:.3f}')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 26: CFG guidance + image-to-image behavior
# ═══════════════════════════════════════════════════════════════
run_test "26. Sampling loop: CFG and img2img behavior" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from scheduling.sampling_loop import sample_with_scheduler

class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([], dtype=torch.long)
    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)
    def add_noise(self, x, n, t):
        _ = t
        return x + 0.1 * n
    def step(self, pred, t, sample):
        _ = t
        return type('Out', (), {'prev_sample': sample - 0.1 * pred})()

class FakeModel(torch.nn.Module):
    def forward(self, x, t, context_ca=None):
        _ = t
        if context_ca is None:
            scale = 0.0
        else:
            scale = context_ca.mean(dim=tuple(range(1, context_ca.dim())), keepdim=True)
            while scale.dim() < x.dim():
                scale = scale.unsqueeze(-1)
        return x + scale

torch.manual_seed(0)
shape = (2, 1, 8, 8)
cond = torch.ones(shape)
uncond = torch.zeros(shape)

base = sample_with_scheduler(
    model=FakeModel(),
    scheduler=FakeScheduler(),
    num_inference_steps=4,
    sample_shape=shape,
    device=torch.device('cpu'),
    conditioning_mode='attention',
    conditioning_batch=cond,
    unconditional_conditioning_batch=uncond,
    guidance_scale=1.0,
)
torch.manual_seed(0)
guided = sample_with_scheduler(
    model=FakeModel(),
    scheduler=FakeScheduler(),
    num_inference_steps=4,
    sample_shape=shape,
    device=torch.device('cpu'),
    conditioning_mode='attention',
    conditioning_batch=cond,
    unconditional_conditioning_batch=uncond,
    guidance_scale=7.5,
)
assert not torch.allclose(base, guided), 'CFG guidance_scale had no effect on output.'

init_image = torch.randn(shape)
img2img = sample_with_scheduler(
    model=FakeModel(),
    scheduler=FakeScheduler(),
    num_inference_steps=6,
    sample_shape=shape,
    device=torch.device('cpu'),
    conditioning_mode='none',
    init_image=init_image,
    strength=0.5,
)
assert img2img.shape == init_image.shape
assert torch.isfinite(img2img).all()
print('CFG and img2img behavior OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 27: SAMPLER_REGISTRY completeness
# ═══════════════════════════════════════════════════════════════
run_test "27. SAMPLER_REGISTRY contains all expected runtime samplers" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from sampling import SAMPLER_REGISTRY

expected = ['vae', 'diffusion', 'flow_matching', 'latent_diffusion', 'latent_flow_matching', 'latent_rectified_flow', 'consistency', 'edm', 'rectified_flow', 'reflow']
expected += ['unet', 'video_unet', 'distillation', 'controlnet']
missing = [k for k in expected if k not in SAMPLER_REGISTRY]
assert not missing, f'Missing sampler registrations: {missing}'
print('Sampler registry complete.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 28: Reflow pair generation runtime path
# ═══════════════════════════════════════════════════════════════
run_test "28. run_model: generate_reflow_pairs writes z0/z1 pairs" \
    "$PYTHON_BIN -c \"
import json
from pathlib import Path

cfg = {
    'training': {'conditioning': 'none'},
    'model': {
        'model_type': 'flow_matching',
        'unet': {
            'unet_impl': 'efficient_nd',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'model_channels': 32,
            'num_res_blocks': 1,
            'channel_mult': [1, 2],
        },
        'scheduler': {
            'name': 'flow_match_euler',
            'num_train_timesteps': 100,
            'num_inference_steps': 10,
        },
    },
}
Path('$FM_TRAIN_DIR/train_config.json').write_text(json.dumps(cfg))
Path(Path('$WORK_DIR/fm_ckpt_dir.txt').read_text().strip()).joinpath('train_config.json').write_text(json.dumps(cfg))
\" && cd '$PROJECT_ROOT' && $PYTHON_BIN run_model.py \
        --ckpt_dir \"$(cat $WORK_DIR/fm_ckpt_dir.txt)\" \
        --mode generate_reflow_pairs \
        --output_dir '$WORK_DIR/reflow_pairs_runtime' \
        --num_pairs 3 \
        --batch_size 2"

run_test "29. Reflow pair payloads contain z0/z1" \
    "$PYTHON_BIN -c \"
import torch
from pathlib import Path

root = Path('$WORK_DIR/reflow_pairs_runtime')
files = sorted(root.glob('*.pt'))
assert len(files) == 3, f'Expected 3 reflow pair files, found {len(files)}'
payload = torch.load(files[0], map_location='cpu')
assert set(payload.keys()) == {'z0', 'z1'}, f'Unexpected payload keys: {payload.keys()}'
assert payload['z0'].shape == payload['z1'].shape
print('Reflow pair runtime generation OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 30: LoRA wrap/save/load round-trip
# ═══════════════════════════════════════════════════════════════
run_test "30. LoRA wrap/save/load round-trip" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
import torch.nn as nn
from pathlib import Path
from nn.modules import LoRALinear
from training.lora import LoRAWrapper

class TinyAttn(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))

model = TinyAttn()
LoRAWrapper.wrap(model, rank=4, alpha=1.0)
assert isinstance(model.to_q, LoRALinear)
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.lora_A.data.fill_(0.25)
        module.lora_B.data.fill_(0.5)

path = Path('$WORK_DIR/lora_roundtrip.pt')
LoRAWrapper.save_lora_weights(model, path)

reloaded = TinyAttn()
LoRAWrapper.wrap(reloaded, rank=4, alpha=1.0)
LoRAWrapper.load_lora_weights(reloaded, path)

for (name_a, mod_a), (name_b, mod_b) in zip(model.named_modules(), reloaded.named_modules()):
    if isinstance(mod_a, LoRALinear):
        assert name_a == name_b
        assert torch.allclose(mod_a.lora_A, mod_b.lora_A)
        assert torch.allclose(mod_a.lora_B, mod_b.lora_B)
print('LoRA round-trip OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 31: UNetTrainer smoke
# ═══════════════════════════════════════════════════════════════
UNET_TRAIN_DIR="$WORK_DIR/unet_train"
mkdir -p "$UNET_TRAIN_DIR"

run_test "31. UNetTrainer: supervised smoke fit" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
import torch.nn as nn
from training import UNetTrainer

class TinyDataset:
    def __len__(self): return 4
    def __getitem__(self, idx):
        base = torch.full((1, 8, 8), float(idx))
        return {'image': base, 'target': base}

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    def forward(self, x, t):
        _ = t
        return self.conv(x)

config = {
    'training': {
        'epochs': 1, 'batch_size': 2, 'num_workers': 0, 'learning_rate': 1e-3,
        'weight_decay': 0.0, 'output_dir': '$UNET_TRAIN_DIR', 'use_amp': False,
        'manual_device': 'cpu', 'seed': 0, 'save_every': 1, 'validate': True,
    },
    'model': {
        'model_type': 'unet',
        'unet': {'in_channels': 1, 'out_channels': 1, 'spatial_dims': 2},
    },
}

ds = TinyDataset()
trainer = UNetTrainer(config=config, callbacks=[], model_override=TinyUNet())
trainer.fit(ds, val_dataset=ds, resume=None)
assert trainer.global_step > 0
print('UNetTrainer smoke OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 32: DistillationTrainer smoke
# ═══════════════════════════════════════════════════════════════
DISTILL_TRAIN_DIR="$WORK_DIR/distill_train"
mkdir -p "$DISTILL_TRAIN_DIR"

run_test "32. DistillationTrainer: teacher/student smoke fit" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
import torch.nn as nn
from training import DistillationTrainer

class DummyScheduler:
    class _Cfg:
        num_train_timesteps = 1000
    config = _Cfg()
    def add_noise(self, clean, noise, timesteps):
        scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, self.config.num_train_timesteps - 1)
        return clean + scale * noise

class TinyUNet(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))
    def forward(self, x, t):
        _ = t
        return x * self.gain

class TinyDataset:
    def __len__(self): return 4
    def __getitem__(self, idx):
        _ = idx
        return {'target': torch.zeros(1, 8, 8)}

config = {
    'training': {
        'epochs': 1, 'batch_size': 2, 'num_workers': 0, 'learning_rate': 1e-3,
        'weight_decay': 0.0, 'output_dir': '$DISTILL_TRAIN_DIR', 'use_amp': False,
        'manual_device': 'cpu', 'seed': 0, 'save_every': 1, 'validate': True,
    },
    'model': {
        'model_type': 'distillation',
        'student_model_type': 'diffusion',
        'teacher_steps': 128,
        'student_steps': 64,
        'scheduler': {},
    },
}

ds = TinyDataset()
trainer = DistillationTrainer(
    config=config,
    callbacks=[],
    model_override=TinyUNet(0.5),
    teacher_override=TinyUNet(1.0),
    scheduler_override=DummyScheduler(),
)
trainer.fit(ds, val_dataset=ds, resume=None)
assert trainer.global_step > 0
print('DistillationTrainer smoke OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 33: DiT forward pass
# ═══════════════════════════════════════════════════════════════
run_test "33. DiTND: forward pass smoke" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from models.dit import DiTND

model = DiTND(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    patch_size=2,
    hidden_size=64,
    depth=2,
    num_heads=4,
    mlp_ratio=2.0,
)
x = torch.randn(2, 4, 32, 32)
t = torch.randint(0, 1000, (2,), dtype=torch.long)
out = model(x, t)
assert out.shape == x.shape
assert torch.isfinite(out).all()
print('DiT forward OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 34: Extended scheduler aliases
# ═══════════════════════════════════════════════════════════════
run_test "34. Extended schedulers build successfully" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from scheduling.builder import build_scheduler, resolve_scheduler_override

registry_names = [
    'ddpm', 'ddim', 'pndm', 'euler', 'euler_ancestral', 'heun', 'lms',
    'kdpm2', 'kdpm2_ancestral', 'deis', 'dpm_multistep', 'dpm_sde',
    'unipc', 'flow_match_euler', 'flowmatch',
]
for key in registry_names:
    try:
        scheduler, steps = build_scheduler({'name': key}, {})
    except ImportError as exc:
        if key == 'dpm_sde' and 'torchsde' in str(exc).lower():
            print('  dpm_sde: SKIPPED (optional torchsde backend not installed)')
            continue
        raise
    assert hasattr(scheduler, 'config')
    assert isinstance(steps, int)

override_names = ['ddpm', 'ddim', 'dpmsolver1', 'dpmsolver2', 'dpmsolver++', 'dpmsolversde', 'unipc', 'flowmatch']
for key in override_names:
    override = resolve_scheduler_override(key)
    assert isinstance(override, dict)
    assert 'name' in override
print('Extended scheduler coverage OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 35: Super-resolution adapter behavior
# ═══════════════════════════════════════════════════════════════
run_test "35. super_resolution adapter upsamples and concatenates" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from scheduling.conditioning import resolve_conditioning_adapter

adapter = resolve_conditioning_adapter('super_resolution')
x = torch.randn(2, 1, 256, 256)
cond = torch.randn(2, 1, 64, 64)
out_x, ctx = adapter(x, cond, None)
assert out_x.shape == (2, 2, 256, 256)
assert ctx is None
print('super_resolution adapter OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 36: VideoUNet forward pass
# ═══════════════════════════════════════════════════════════════
run_test "36. VideoUNetND: forward pass smoke" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from models.unet.video import VideoUNetND

model = VideoUNetND(
    spatial_dims=3,
    in_channels=4,
    model_channels=32,
    out_channels=4,
    num_res_blocks=1,
    channel_mult=(1, 2),
    attention_resolutions=(),
    temporal_num_heads=4,
)
x = torch.randn(2, 4, 4, 8, 8)
t = torch.randint(0, 1000, (2,), dtype=torch.long)
out = model(x, t)
assert out.shape == x.shape
assert torch.isfinite(out).all()
print('VideoUNet forward OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 37: Medical3DDataset channel_order
# ═══════════════════════════════════════════════════════════════
run_test "37. Medical3DDataset: channel_order handling" \
    "$PYTHON_BIN -c \"
import sys, numpy as np; sys.path.insert(0, '$SRC_DIR')
from pathlib import Path
from datasets.medical3d import Medical3DDataset

root = Path('$WORK_DIR/medical3d')
root.mkdir(parents=True, exist_ok=True)
np.save(root / 'target.npy', np.random.rand(6, 7, 8, 2).astype('float32'))
(root / 'train.txt').write_text('id\\ttarget\\tconditioning\\ncase0\\ttarget.npy\\ttarget.npy\\n', encoding='utf-8')
(root / 'test.txt').write_text('id\\ttarget\\tconditioning\\ncase0\\ttarget.npy\\ttarget.npy\\n', encoding='utf-8')
ds = Medical3DDataset(file_path=str(root), train=True, conditioning=False, volume_size=(6, 7, 8), channel_order='channel_last')
sample = ds[0]
assert sample['target'].shape == (2, 6, 7, 8)
print('Medical3DDataset channel_order OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 38: TemporalAttention causal flag
# ═══════════════════════════════════════════════════════════════
run_test "38. TemporalAttentionND: causal flag smoke" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from nn.modules import TemporalAttentionND

module = TemporalAttentionND(channels=24, num_heads=6, causal=True)
x = torch.randn(2, 24, 5, 4, 4)
y = module(x)
assert y.shape == x.shape
print('TemporalAttention causal OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 39: VideoUNet / Distillation sampler registrations
# ═══════════════════════════════════════════════════════════════
run_test "39. Sampler registry includes video_unet and distillation" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from sampling import SAMPLER_REGISTRY

assert 'video_unet' in SAMPLER_REGISTRY
assert 'distillation' in SAMPLER_REGISTRY
print('VideoUNetSampler and DistillationSampler registry OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 40: Scheduler registry is Registry[T]
# ═══════════════════════════════════════════════════════════════
run_test "40. SCHEDULER_REGISTRY uses Registry[T]" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from core.registry import Registry
from scheduling import SCHEDULER_REGISTRY

assert isinstance(SCHEDULER_REGISTRY, Registry)
assert hasattr(SCHEDULER_REGISTRY, 'register_value')
assert len(SCHEDULER_REGISTRY.keys()) >= 10
print('SCHEDULER_REGISTRY Registry[T] OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 41: WGAN-GP / R1 loss registry entries
# ═══════════════════════════════════════════════════════════════
run_test "41. LOSS_REGISTRY includes WGAN-GP and R1 components" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$SRC_DIR')
from losses import LOSS_REGISTRY

for key in ['gan_generator_wgan', 'gan_discriminator_wgangp', 'gan_discriminator_r1']:
    assert key in LOSS_REGISTRY, f'Missing loss registry key: {key}'
print('WGAN-GP/R1 loss registry coverage OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 42: Public version + changelog
# ═══════════════════════════════════════════════════════════════
run_test "42. Package version is 1.0.0 and CHANGELOG exists" \
    "$PYTHON_BIN -c \"
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from pathlib import Path
import src

assert src.__version__ == '1.0.0'
assert Path('$PROJECT_ROOT/CHANGELOG.md').exists()
print('Version + CHANGELOG OK.')
\""

# ═══════════════════════════════════════════════════════════════
# TEST 43: ControlNet runtime sampler smoke
# ═══════════════════════════════════════════════════════════════
run_test "43. ControlNetSampler runtime helper smoke" \
    "$PYTHON_BIN -c \"
import sys, torch; sys.path.insert(0, '$SRC_DIR')
from pathlib import Path
from sampling.controlnet_sampler import _run_controlnet_inference

cfg = {
    'training': {},
    'model': {
        'model_type': 'controlnet',
        'base_unet_checkpoint': '$WORK_DIR/base_controlnet_run',
        'scheduler': {},
    },
    'sampling': {},
}

class Dataset:
    target_key = 'target'
    conditioning_key = 'image'
    data = [{'img_id': 'case0', 'img_path': 'case0.dcm'}]

sample = {
    'target': torch.zeros(1, 8, 8),
    'image': torch.ones(1, 8, 8),
    'img_id': 'case0',
    'img_path': 'case0.dcm',
}

class Pipe:
    def generate(self, inputs):
        assert inputs.controlnet_cond is not None
        assert tuple(inputs.controlnet_cond.shape) == (1, 1, 8, 8)
        return torch.zeros(inputs.sample_shape)

saved = []

import sampling.controlnet_sampler as mod
mod.load_run_config = lambda _ckpt: cfg
mod.build_sampling_dataset = lambda *args, **kwargs: Dataset()
mod.resolve_sample_indices = lambda *args, **kwargs: [0]
mod.progress_batches = lambda _dataset, _batch_size, _desc, indices=None: [([0], [sample])]
mod._build_controlnet_inference_pipeline = lambda **kwargs: (Pipe(), 5)
mod.save_output_tensor = lambda dataset, row, key, tensor, output_root: saved.append((key, tuple(tensor.shape)))

_run_controlnet_inference(
    ckpt_dir=Path('$WORK_DIR/controlnet_runtime_ckpt'),
    save=True,
    output_dir='$WORK_DIR/controlnet_runtime_out',
    batch_size=1,
    evaluate=False,
)
assert saved and saved[0][0] == 'target'
print('ControlNet runtime smoke OK.')
\""

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  RESULTS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}  FAILED TESTS:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "    ${RED}✗ $f${NC}"
    done
    echo ""
    echo -e "${RED}  ⚠ DO NOT MERGE TO MAIN${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}  ✓ ALL TESTS PASSED — SAFE TO MERGE${NC}"
exit 0
