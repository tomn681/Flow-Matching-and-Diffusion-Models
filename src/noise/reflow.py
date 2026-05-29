from __future__ import annotations

from pathlib import Path

import torch

from core.types import NoisyBatch
from scheduling import sample_with_scheduler
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
) -> None:
    """Generate and persist (z0, z1) coupling pairs for reflow training.

    `sample_shape` is per-sample shape (channels + spatial dims), without batch dim.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not sample_shape or any(dim <= 0 for dim in sample_shape):
        raise ValueError("sample_shape must contain positive per-sample dimensions.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()

    written = 0
    with torch.no_grad():
        while written < num_pairs:
            current_bs = min(batch_size, num_pairs - written)
            current_shape = (current_bs, *sample_shape)
            z0 = torch.randn(current_shape, device=device)
            z1 = sample_with_scheduler(
                model=model,
                scheduler=scheduler,
                num_inference_steps=int(num_inference_steps),
                sample_shape=current_shape,
                device=device,
                conditioning_mode="none",
                init_sample=z0,
            )
            for idx in range(current_bs):
                payload = {
                    "z0": z0[idx].detach().cpu().contiguous(),
                    "z1": z1[idx].detach().cpu().contiguous(),
                }
                torch.save(payload, out_root / f"{written:08d}.pt")
                written += 1

    if model_was_training:
        model.train()


@NOISE_REGISTRY.register("reflow")
class ReflowNoise:
    """Reflow noise process using pre-generated (z0, z1) coupling pairs."""

    def __init__(self, scheduler, *, pairs_dir: str) -> None:
        self.scheduler = scheduler
        self.pairs_dir = Path(pairs_dir)
        if not self.pairs_dir.exists():
            raise FileNotFoundError(f"Reflow pairs directory not found: {self.pairs_dir}")
        self._pairs = self._load_pairs(self.pairs_dir)

    def _load_pairs(self, root: Path) -> list[dict[str, torch.Tensor]]:
        paths = sorted(root.glob("*.pt"))
        if not paths:
            raise ValueError(f"No reflow pair files found under: {root}")
        pairs: list[dict[str, torch.Tensor]] = []
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict) or "z0" not in payload or "z1" not in payload:
                raise ValueError(f"Invalid reflow pair file: {path}")
            z0 = torch.as_tensor(payload["z0"]).float().contiguous()
            z1 = torch.as_tensor(payload["z1"]).float().contiguous()
            if tuple(z0.shape) != tuple(z1.shape):
                raise ValueError(f"Mismatched z0/z1 shapes in {path}: {tuple(z0.shape)} vs {tuple(z1.shape)}")
            pairs.append({"z0": z0, "z1": z1})
        return pairs

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        batch = clean.size(0)
        idx = torch.randint(0, len(self._pairs), (batch,), device=device)
        z0 = torch.stack([self._pairs[int(i)]["z0"] for i in idx.tolist()], dim=0).to(device)
        z1 = torch.stack([self._pairs[int(i)]["z1"] for i in idx.tolist()], dim=0).to(device)

        if tuple(z0.shape[1:]) != tuple(clean.shape[1:]):
            raise ValueError(
                f"Reflow pair tensor shape {tuple(z0.shape[1:])} does not match training batch shape {tuple(clean.shape[1:])}."
            )

        t = torch.rand(batch, device=device)
        t_view = t.view(batch, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * z0 + t_view * z1
        target = z1 - z0
        timesteps = (t * (self.scheduler.config.num_train_timesteps - 1)).long()
        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)


__all__ = ["ReflowNoise", "generate_reflow_pairs"]
