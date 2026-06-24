"""Canonical packaged training entrypoint wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_impl():
    repo_root = Path(__file__).resolve().parent.parent
    impl_path = repo_root / "train.py"
    spec = importlib.util.spec_from_file_location("genlib._train_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training implementation from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

_build_parser = _impl._build_parser
_merge_overrides = _impl._merge_overrides
main = _impl.main

__all__ = ["main", "_build_parser", "_merge_overrides"]
