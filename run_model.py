"""Compatibility wrapper for `python run_model.py` from repository root."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src import run_model as _impl

# Re-export hook points used by unit tests.
load_run_config = _impl.load_run_config
SAMPLER_REGISTRY = _impl.SAMPLER_REGISTRY
_supports_mode = _impl._supports_mode
_interrupt_label = _impl._interrupt_label
_exit_on_keyboard_interrupt = _impl._exit_on_keyboard_interrupt


def main() -> None:
    """Delegate to src.run_model while honoring monkeypatches on this wrapper."""
    _impl.load_run_config = load_run_config
    _impl.SAMPLER_REGISTRY = SAMPLER_REGISTRY
    _impl._supports_mode = _supports_mode
    _impl._interrupt_label = _interrupt_label
    _impl._exit_on_keyboard_interrupt = _exit_on_keyboard_interrupt
    _impl.main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _exit_on_keyboard_interrupt("sample")
