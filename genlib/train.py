"""Canonical packaged training entrypoint wrapper."""

from __future__ import annotations

from src import train as _impl

_build_parser = _impl._build_parser
_merge_overrides = _impl._merge_overrides
main = _impl.main

__all__ = ["main", "_build_parser", "_merge_overrides"]

