from __future__ import annotations

import bisect
import itertools
import logging
from pathlib import Path
import re
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

import utils
from core.types import NoisyBatch
from scheduling import sample_with_scheduler
from core.noise_contracts import validate_noise_scheduler_contract
from .base import BaseNoiseProcess
from .registry import NOISE_REGISTRY

_UNCONDITIONED_MODES = {"none", "false", "off"}
_SHARD_NAME_RE = re.compile(r"^pairs_(?P<start>\d{8})_n(?P<count>\d+)\.pt$")
_LEGACY_PAIR_RE = re.compile(r"^\d+\.pt$")


@dataclass(frozen=True)
class _PairRecord:
    path: Path
    count: int


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
    conditioning_dataset=None,
    loader_workers: int = 0,
    pairs_per_file: int = 256,
) -> None:
    """Generate and persist (z0, z1) coupling pairs for reflow training.

    `sample_shape` is per-sample shape (channels + spatial dims), without batch dim.

    When `conditioning_dataset` is provided and `conditioning_mode` is active,
    pairs are generated conditionally: each z1 = F_θ(z0 | LDCT_i) using a real
    LDCT image, and the conditioning tensor is saved alongside (z0, z1) so that
    reflow training can use the same conditioning that produced each trajectory.

    When `conditioning_dataset` is None and `conditioning_mode` is "concatenate",
    null (zero) conditioning is used — this is the unconditional reflow path and
    does NOT preserve anatomical correspondence.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if loader_workers < 0:
        raise ValueError("loader_workers must be >= 0")
    if pairs_per_file <= 0:
        raise ValueError("pairs_per_file must be > 0")
    if not sample_shape or any(dim <= 0 for dim in sample_shape):
        raise ValueError("sample_shape must contain positive per-sample dimensions.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    validate_noise_scheduler_contract("reflow", scheduler)

    _cond_mode = str(conditioning_mode or "none").strip().lower()
    _conditional = _cond_mode not in _UNCONDITIONED_MODES and conditioning_dataset is not None
    _use_concat_null = _cond_mode == "concatenate" and not _conditional

    # Build cycling dataset iterator for conditional generation
    _data_iter = None
    if _conditional:
        _loader = DataLoader(
            conditioning_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=int(loader_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(loader_workers > 0),
        )
        _data_iter = itertools.cycle(_loader)
        cache_root = getattr(conditioning_dataset, "cache_root", None)
        logging.info(
            "Reflow pair generation dataset: samples=%d | cache=%s | loader_workers=%d | pairs_per_file=%d",
            len(conditioning_dataset),
            str(cache_root) if cache_root is not None else "<none>",
            int(loader_workers),
            int(pairs_per_file),
        )
    else:
        logging.info(
            "Reflow pair generation: unconditional | loader_workers=%d | pairs_per_file=%d",
            int(loader_workers),
            int(pairs_per_file),
        )

    model_was_training = model.training
    model.eval()

    try:
        from tqdm import tqdm  # type: ignore
        _pbar = tqdm(total=num_pairs, desc="generate_reflow_pairs", unit="pair", dynamic_ncols=True)
    except Exception:
        _pbar = None

    existing = sorted(out_root.glob("*.pt"))
    written = sum(_saved_pair_count(path) for path in existing)
    if written > 0:
        print(f"Resuming pair generation: {written}/{num_pairs} pairs already exist, skipping.", flush=True)
        if _pbar is not None:
            _pbar.update(written)

    shard_buffers: dict[str, list[torch.Tensor]] = {"z0": [], "z1": [], "cond": []}
    shard_fill = 0

    def _append_to_shards(z0_batch: torch.Tensor, z1_batch: torch.Tensor, cond_batch: torch.Tensor | None) -> int:
        nonlocal shard_fill, written
        offset = 0
        total = int(z0_batch.size(0))
        while offset < total:
            take = min(int(pairs_per_file - shard_fill), total - offset)
            shard_buffers["z0"].append(z0_batch[offset : offset + take].detach().cpu().contiguous())
            shard_buffers["z1"].append(z1_batch[offset : offset + take].detach().cpu().contiguous())
            if cond_batch is not None:
                shard_buffers["cond"].append(cond_batch[offset : offset + take].detach().cpu().contiguous())
            shard_fill += take
            offset += take
            if shard_fill >= pairs_per_file:
                _flush_shard()
        return total

    def _flush_shard() -> None:
        nonlocal shard_fill, written
        if shard_fill <= 0:
            return
        start_index = written
        payload = {
            "format": "reflow_pair_shard_v1",
            "start_index": int(start_index),
            "count": int(shard_fill),
            "z0": torch.cat(shard_buffers["z0"], dim=0),
            "z1": torch.cat(shard_buffers["z1"], dim=0),
        }
        if shard_buffers["cond"]:
            payload["cond"] = torch.cat(shard_buffers["cond"], dim=0)
        shard_path = out_root / f"pairs_{start_index:08d}_n{shard_fill:06d}.pt"
        torch.save(payload, shard_path)
        written += int(shard_fill)
        shard_buffers["z0"].clear()
        shard_buffers["z1"].clear()
        shard_buffers["cond"].clear()
        shard_fill = 0

    with torch.no_grad():
        while written + shard_fill < num_pairs:
            current_bs = min(batch_size, num_pairs - written - shard_fill)
            current_shape = (current_bs, *sample_shape)
            z0 = torch.randn(current_shape, device=device)

            if _conditional:
                data_batch = next(_data_iter)
                cond_batch = data_batch["image"].to(device)
                if written == 0:
                    logging.info(
                        "Reflow pair first batch: sample_shape=%s | cond_shape=%s | device=%s",
                        tuple(z0.shape),
                        tuple(cond_batch.shape),
                        str(device),
                    )
                # Trim to current_bs in case the loader returned a larger batch
                cond_batch = cond_batch[:current_bs]
                # If the loader batch was smaller, pad by repeating last row
                if cond_batch.size(0) < current_bs:
                    pad = cond_batch[-1:].expand(current_bs - cond_batch.size(0), *cond_batch.shape[1:])
                    cond_batch = torch.cat([cond_batch, pad], dim=0)
                z1 = sample_with_scheduler(
                    model=model,
                    scheduler=scheduler,
                    num_inference_steps=int(num_inference_steps),
                    sample_shape=current_shape,
                    device=device,
                    conditioning_mode=_cond_mode,
                    conditioning_batch=cond_batch,
                    init_sample=z0,
                )
                _append_to_shards(z0, z1, cond_batch)
            else:
                null_cond = torch.zeros_like(z0) if _use_concat_null else None
                if written == 0:
                    logging.info(
                        "Reflow pair first batch: sample_shape=%s | cond_shape=%s | device=%s",
                        tuple(z0.shape),
                        tuple(null_cond.shape) if null_cond is not None else None,
                        str(device),
                    )
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
                _append_to_shards(z0, z1, null_cond if _use_concat_null else None)

            if _pbar is not None:
                _pbar.update(current_bs)

        _flush_shard()

    if _pbar is not None:
        _pbar.close()

    if model_was_training:
        model.train()


@NOISE_REGISTRY.register("reflow")
class ReflowNoise(BaseNoiseProcess):
    """Reflow noise process using pre-generated (z0, z1) coupling pairs.

    If pairs were generated conditionally (with a saved "cond" tensor), that
    conditioning is loaded and returned in NoisyBatch.extra["conditioning"] so
    that GenerativeTrainer._run_step can override the dataset conditioning with
    the pair-specific conditioning used during generation.
    """

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
        super().__init__(scheduler)
        validate_noise_scheduler_contract("reflow", scheduler)
        self.pairs_dir = Path(pairs_dir)
        if not self.pairs_dir.exists():
            raise FileNotFoundError(f"Reflow pairs directory not found: {self.pairs_dir}")
        self._pair_records, self._pair_counts = self._index_pairs(self.pairs_dir)
        self._num_pairs = int(self._pair_counts[-1])
        self.timestep_sampling = str(timestep_sampling).strip().lower()
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.shift = float(
            shift
            if shift is not None
            else getattr(getattr(scheduler, "config", None), "shift", 1.0) or 1.0
        )
        self._payload_cache: OrderedDict[Path, dict] = OrderedDict()
        self._payload_cache_size = 4

    def _index_pairs(self, root: Path) -> tuple[list[_PairRecord], list[int]]:
        paths = sorted(root.glob("*.pt"))
        if not paths:
            raise ValueError(f"No reflow pair files found under: {root}")
        records: list[_PairRecord] = []
        counts: list[int] = []
        total = 0
        for path in paths:
            count = _saved_pair_count(path)
            if count <= 0:
                raise ValueError(f"Invalid reflow pair file count for {path}")
            records.append(_PairRecord(path=path, count=int(count)))
            total += int(count)
            counts.append(total)
        return records, counts

    def _load_payload(self, path: Path) -> dict:
        cached = self._payload_cache.get(path)
        if cached is not None:
            self._payload_cache.move_to_end(path)
            return cached
        payload = utils.safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid reflow pair file: {path}")
        self._payload_cache[path] = payload
        self._payload_cache.move_to_end(path)
        while len(self._payload_cache) > self._payload_cache_size:
            self._payload_cache.popitem(last=False)
        return payload

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
        indices = torch.randint(0, self._num_pairs, (batch,), device=device).tolist()
        grouped: dict[int, list[tuple[int, int]]] = {}
        for batch_slot, global_idx in enumerate(indices):
            record_idx = bisect.bisect_right(self._pair_counts, int(global_idx))
            previous = 0 if record_idx == 0 else self._pair_counts[record_idx - 1]
            local_idx = int(global_idx) - int(previous)
            grouped.setdefault(record_idx, []).append((batch_slot, local_idx))

        z0_slots: list[torch.Tensor | None] = [None] * batch
        z1_slots: list[torch.Tensor | None] = [None] * batch
        cond_slots: list[torch.Tensor | None] = [None] * batch

        for record_idx, requests in grouped.items():
            record = self._pair_records[int(record_idx)]
            payload = self._load_payload(record.path)
            if "z0" not in payload or "z1" not in payload:
                raise ValueError(f"Invalid reflow pair file: {record.path}")
            z0_tensor = torch.as_tensor(payload["z0"]).float().contiguous()
            z1_tensor = torch.as_tensor(payload["z1"]).float().contiguous()
            if record.count == 1:
                if tuple(z0_tensor.shape) != tuple(z1_tensor.shape):
                    raise ValueError(
                        f"Mismatched z0/z1 shapes in {record.path}: {tuple(z0_tensor.shape)} vs {tuple(z1_tensor.shape)}"
                    )
                cond_tensor = (
                    torch.as_tensor(payload["cond"]).float().contiguous()
                    if "cond" in payload
                    else None
                )
                for batch_slot, _local_idx in requests:
                    z0_slots[batch_slot] = z0_tensor
                    z1_slots[batch_slot] = z1_tensor
                    if cond_tensor is not None:
                        cond_slots[batch_slot] = cond_tensor
                continue

            if z0_tensor.size(0) != record.count or z1_tensor.size(0) != record.count:
                raise ValueError(
                    f"Shard count mismatch in {record.path}: expected {record.count}, got "
                    f"{int(z0_tensor.size(0))}/{int(z1_tensor.size(0))}"
                )
            if tuple(z0_tensor.shape) != tuple(z1_tensor.shape):
                raise ValueError(
                    f"Mismatched z0/z1 shard shapes in {record.path}: {tuple(z0_tensor.shape)} vs {tuple(z1_tensor.shape)}"
                )
            cond_tensor = (
                torch.as_tensor(payload["cond"]).float().contiguous()
                if "cond" in payload
                else None
            )
            if cond_tensor is not None and cond_tensor.size(0) != record.count:
                raise ValueError(
                    f"Conditioning shard count mismatch in {record.path}: expected {record.count}, got {int(cond_tensor.size(0))}"
                )
            for batch_slot, local_idx in requests:
                z0_slots[batch_slot] = z0_tensor[local_idx]
                z1_slots[batch_slot] = z1_tensor[local_idx]
                if cond_tensor is not None:
                    cond_slots[batch_slot] = cond_tensor[local_idx]

        if any(tensor is None for tensor in z0_slots) or any(tensor is None for tensor in z1_slots):
            raise RuntimeError("Failed to resolve all sampled reflow pairs.")
        z0 = torch.stack([tensor for tensor in z0_slots if tensor is not None], dim=0).to(device)
        z1 = torch.stack([tensor for tensor in z1_slots if tensor is not None], dim=0).to(device)

        if tuple(z0.shape[1:]) != tuple(clean.shape[1:]):
            raise ValueError(
                f"Reflow pair tensor shape {tuple(z0.shape[1:])} does not match training batch shape {tuple(clean.shape[1:])}."
            )

        t = self._sample_t(batch, device)
        t_view = t.view(batch, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - t_view) * z1 + t_view * z0
        target = z0 - z1
        timesteps = t * float(self.scheduler.config.num_train_timesteps - 1)

        extra: dict = {}
        if all(tensor is not None for tensor in cond_slots):
            extra["conditioning"] = torch.stack([tensor for tensor in cond_slots if tensor is not None], dim=0).to(device)
        elif any(tensor is not None for tensor in cond_slots):
            raise RuntimeError("Partial conditioning recovery detected in reflow pair batch.")

        return NoisyBatch(noisy=noisy, target=target, timesteps=timesteps, extra=extra)


def _saved_pair_count(path: Path) -> int:
    match = _SHARD_NAME_RE.match(path.name)
    if match is not None:
        return int(match.group("count"))
    if _LEGACY_PAIR_RE.match(path.name):
        return 1
    payload = utils.safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid reflow pair payload: {path}")
    if "count" in payload:
        return int(payload["count"])
    return 1


__all__ = ["ReflowNoise", "generate_reflow_pairs"]
