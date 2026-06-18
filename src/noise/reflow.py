from __future__ import annotations

from pathlib import Path

import torch

import utils
from core.types import NoisyBatch
from scheduling import sample_with_scheduler
from core.noise_contracts import validate_noise_scheduler_contract
from .registry import NOISE_REGISTRY


def generate_reflow_pairs(
    *,
    model: torch.nn.Module,
    scheduler,
    num_pairs: int,
    sample_shape: tuple[int, ...],
    device: torch.device,
    output_dir: str | Path,
    num_inference_steps: int,
    batch_size: int = 4,
    conditioning_mode: str = "none",
) -> None:
    """Generate and persist (z0, z1) coupling pairs for reflow training.

    `sample_shape` is per-sample shape (channels + spatial dims), without batch dim.
    For concatenate-conditioned models, null (zero) conditioning is used so that the
    model receives the correct number of input channels.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not sample_shape or any(dim <= 0 for dim in sample_shape):
        raise ValueError("sample_shape must contain positive per-sample dimensions.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    validate_noise_scheduler_contract("reflow", scheduler)

    _cond_mode = str(conditioning_mode or "none").strip().lower()
    _use_concat_null = _cond_mode == "concatenate"

    model_was_training = model.training
    model.eval()

    try:
        from tqdm import tqdm  # type: ignore
        _pbar = tqdm(total=num_pairs, desc="generate_reflow_pairs", unit="pair", dynamic_ncols=True)
    except Exception:
        _pbar = None

    written = 0
    with torch.no_grad():
        while written < num_pairs:
            current_bs = min(batch_size, num_pairs - written)
            current_shape = (current_bs, *sample_shape)
            z0 = torch.randn(current_shape, device=device)
            null_cond = torch.zeros_like(z0) if _use_concat_null else None
            z1 = sample_with_scheduler(
                model=model,
                scheduler=scheduler,
                num_inference_steps=int(num_inference_steps),
                sample_shape=current_shape,
                device=device,
                conditioning_mode=_cond_mode if _use_concat_null else "none",
                conditioning_batch=null_cond,
                init_sample=z0,
            )
            for idx in range(current_bs):
                payload = {
                    "z0": z0[idx].detach().cpu().contiguous(),
                    "z1": z1[idx].detach().cpu().contiguous(),
                }
                torch.save(payload, out_root / f"{written:08d}.pt")
                written += 1
            if _pbar is not None:
                _pbar.update(current_bs)

    if _pbar is not None:
        _pbar.close()

    if model_was_training:
        model.train()


@NOISE_REGISTRY.register("reflow")
class ReflowNoise:
    """Reflow noise process using pre-generated (z0, z1) coupling pairs."""

    def __init__(
        self,
        scheduler,
        *,
        pairs_dir: str,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        shift: float | None = None,
    ) -> None:
        self.scheduler = scheduler
        validate_noise_scheduler_contract("reflow", scheduler)
        self.pairs_dir = Path(pairs_dir)
        if not self.pairs_dir.exists():
            raise FileNotFoundError(f"Reflow pairs directory not found: {self.pairs_dir}")
        self._pair_paths = self._index_pairs(self.pairs_dir)
        self.timestep_sampling = str(timestep_sampling).strip().lower()
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.shift = float(
            shift
            if shift is not None
            else getattr(getattr(scheduler, "config", None), "shift", 1.0) or 1.0
        )

    def _index_pairs(self, root: Path) -> list[Path]:
        paths = sorted(root.glob("*.pt"))
        if not paths:
            raise ValueError(f"No reflow pair files found under: {root}")
        return paths

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.timestep_sampling in {"logit_normal", "lognormal_logit"}:
            normal = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(normal)
        else:
            raise ValueError(
                f"Unsupported reflow timestep_sampling '{self.timestep_sampling}'. "
                "Expected one of {'uniform', 'logit_normal'}."
            )
        if self.shift > 0.0 and self.shift != 1.0:
            t = (self.shift * t) / (1.0 + (self.shift - 1.0) * t)
        return t.clamp(1e-5, 1.0 - 1e-5)

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        batch = clean.size(0)
        indices = torch.randint(0, len(self._pair_paths), (batch,), device=device).tolist()
        z0_list: list[torch.Tensor] = []
        z1_list: list[torch.Tensor] = []
        for idx in indices:
            pair_path = self._pair_paths[int(idx)]
            payload = utils.safe_torch_load(pair_path, map_location="cpu")
            if not isinstance(payload, dict) or "z0" not in payload or "z1" not in payload:
                raise ValueError(f"Invalid reflow pair file: {pair_path}")
            z0_tensor = torch.as_tensor(payload["z0"]).float().contiguous()
            z1_tensor = torch.as_tensor(payload["z1"]).float().contiguous()
            if tuple(z0_tensor.shape) != tuple(z1_tensor.shape):
                raise ValueError(
                    f"Mismatched z0/z1 shapes in {pair_path}: {tuple(z0_tensor.shape)} vs {tuple(z1_tensor.shape)}"
                )
            z0_list.append(z0_tensor)
            z1_list.append(z1_tensor)

        z0 = torch.stack(z0_list, dim=0).to(device)
        z1 = torch.stack(z1_list, dim=0).to(device)

        if tuple(z0.shape[1:]) != tuple(clean.shape[1:]):
            raise ValueError(
                f"Reflow pair tensor shape {tuple(z0.shape[1:])} does not match training batch shape {tuple(clean.shape[1:])}."
            )

        t = self._sample_t(batch, device)
        t_view = t.view(batch, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * z1 + t_view * z0
        target = z0 - z1
        timesteps = t * float(self.scheduler.config.num_train_timesteps - 1)
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)


__all__ = ["ReflowNoise", "generate_reflow_pairs"]
