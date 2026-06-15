"""Canonical packaged runtime entrypoint wrapper."""

from __future__ import annotations

from src import run_model as _impl

from genlib.sampling import CheckpointResolver, SAMPLER_REGISTRY, SamplingEngine, SamplingRequest

load_run_config = _impl.load_run_config
_supports_mode = _impl._supports_mode
_interrupt_label = _impl._interrupt_label
_exit_on_keyboard_interrupt = _impl._exit_on_keyboard_interrupt
_build_parser = _impl._build_parser


def main(argv: list[str] | None = None) -> None:
    _impl.load_run_config = load_run_config
    _impl._supports_mode = _supports_mode
    _impl._interrupt_label = _interrupt_label
    _impl._exit_on_keyboard_interrupt = _exit_on_keyboard_interrupt
    _impl.main(argv)


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
