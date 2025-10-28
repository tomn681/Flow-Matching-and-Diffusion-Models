"""
Convenience script to execute the embedded self-tests across the diffusion
library. This allows running the same coverage that is available when invoking
each module directly while keeping imports package-friendly.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

# Ensure both the package root (`src`) and top-level package (`src.*`) are importable.
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MODULES_WITH_TESTS = [
    "src.nn.blocks.attention",
    "src.nn.blocks.residual",
    "src.nn.ops.pooling",
    "src.nn.ops.upsampling",
    "src.models.unet.unet",
]


def main() -> int:
    failures: list[tuple[str, BaseException]] = []

    for module_name in MODULES_WITH_TESTS:
        print(f"\n=== Running self-tests for {module_name} ===")
        mod = importlib.import_module(module_name)
        test_hook = getattr(mod, "run_self_tests", None)
        if test_hook is None:
            print(f"{module_name}: no self-test hook found, skipping.")
            continue

        try:
            test_hook()
        except Exception as exc:  # pragma: no cover - manual smoke harness
            failures.append((module_name, exc))
            traceback.print_exc()

    if failures:
        print("\nSelf-tests finished with failures:")
        for name, exc in failures:
            print(f" - {name}: {exc}")
        return 1

    print("\nAll module self-tests completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
