"""Canonical packaged runtime entrypoint wrapper."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def _load_impl():
    repo_root = Path(__file__).resolve().parent.parent
    impl_path = repo_root / "src" / "run_model.py"
    spec = importlib.util.spec_from_file_location("genlib._run_model_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load runtime implementation from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = None


def _get_impl():
    global _IMPL
    if _IMPL is None:
        _IMPL = _load_impl()
    return _IMPL


def load_run_config(*args, **kwargs):
    return _get_impl().load_run_config(*args, **kwargs)


def _supports_mode(*args, **kwargs):
    return _get_impl()._supports_mode(*args, **kwargs)


def _interrupt_label(*args, **kwargs):
    return _get_impl()._interrupt_label(*args, **kwargs)


def _exit_on_keyboard_interrupt(*args, **kwargs):
    return _get_impl()._exit_on_keyboard_interrupt(*args, **kwargs)


def _build_parser(*args, **kwargs):
    return _get_impl()._build_parser(*args, **kwargs)


def main(argv: list[str] | None = None) -> None:
    impl = _get_impl()
    impl.main(argv)


def __getattr__(name: str):
    if name in {"CheckpointResolver", "SAMPLER_REGISTRY", "SamplingEngine", "SamplingRequest"}:
        sampling_mod = importlib.import_module("sampling")
        return getattr(sampling_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckpointResolver",
    "SAMPLER_REGISTRY",
    "SamplingEngine",
    "SamplingRequest",
    "_build_parser",
    "_exit_on_keyboard_interrupt",
    "_interrupt_label",
    "_supports_mode",
    "load_run_config",
    "main",
]
