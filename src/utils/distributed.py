from __future__ import annotations

import os

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - optional dependency
    dist = None


def _dist_available() -> bool:
    return dist is not None and dist.is_available()


def setup_distributed(backend: str | None = None) -> bool:
    if not _dist_available():
        return False
    if dist.is_initialized():
        return True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    backend = backend or ("nccl" if torch and torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    return True


def is_distributed() -> bool:
    return _dist_available() and dist.is_initialized()


def is_main_process() -> bool:
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    if not is_distributed():
        return 0
    return int(dist.get_rank())


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return int(dist.get_world_size())


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_object(value, *, src: int = 0):
    if not is_distributed():
        return value
    payload = [value if get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_reduce_tensor(tensor, *, op=None):
    if not is_distributed():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=op or dist.ReduceOp.SUM)
    return reduced


def all_reduce_mean(tensor):
    if not is_distributed():
        return tensor
    reduced = all_reduce_tensor(tensor)
    return reduced / float(get_world_size())
