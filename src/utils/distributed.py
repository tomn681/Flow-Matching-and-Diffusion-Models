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

try:  # pragma: no cover - optional dependency surface
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType
except Exception:  # pragma: no cover
    FullStateDictConfig = None
    FullyShardedDataParallel = None
    StateDictType = None


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


def fsdp_available() -> bool:
    return FullyShardedDataParallel is not None


def wrap_fsdp(module, *, device: torch.device | None = None):
    if not fsdp_available():
        raise RuntimeError("FSDP requested but torch.distributed.fsdp is unavailable in this environment.")
    kwargs = {}
    if device is not None and getattr(device, "type", None) == "cuda":
        kwargs["device_id"] = device
    return FullyShardedDataParallel(module, **kwargs)


def is_fsdp_module(module) -> bool:
    return fsdp_available() and isinstance(module, FullyShardedDataParallel)


def fsdp_full_state_dict(module) -> dict:
    if not is_fsdp_module(module):
        return module.state_dict()
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
        return module.state_dict()


def fsdp_load_full_state_dict(module, state_dict: dict) -> None:
    if not is_fsdp_module(module):
        module.load_state_dict(state_dict)
        return
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
        module.load_state_dict(state_dict)
