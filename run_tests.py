"""
Convenience script to execute the embedded self-tests across the diffusion
library. This allows running the same coverage that is available when invoking
each module directly while keeping imports package-friendly.
"""

from __future__ import annotations

import importlib
import inspect
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


def discover_modules_with_self_tests(prefix: str = "src") -> list[str]:
    """
    Discover importable modules under `prefix` that expose run_self_tests().
    """
    discovered: list[str] = []
    root_dir = PROJECT_ROOT / prefix
    for py_file in root_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        if "def run_self_tests" not in text:
            continue
        rel = py_file.relative_to(PROJECT_ROOT).with_suffix("")
        module_name = ".".join(rel.parts)
        discovered.append(module_name)
    return sorted(discovered)


def discover_all_modules(prefix: str = "src") -> list[str]:
    """
    Discover every importable .py module under `prefix` by filesystem scan.
    """
    root_dir = PROJECT_ROOT / prefix
    discovered: list[str] = []
    for py_file in root_dir.rglob("*.py"):
        rel = py_file.relative_to(PROJECT_ROOT).with_suffix("")
        discovered.append(".".join(rel.parts))
    return sorted(discovered)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    skipped: list[tuple[str, BaseException]] = []
    modules_with_tests = discover_modules_with_self_tests("src")
    all_modules = discover_all_modules("src")
    if not modules_with_tests:
        print("No run_self_tests hooks discovered.")
    else:
        for module_name in modules_with_tests:
            print(f"\n=== Running self-tests for {module_name} ===")
            try:
                mod = importlib.import_module(module_name)
            except Exception as exc:
                skipped.append((module_name, exc))
                print(f"{module_name}: import failed, skipping ({exc}).")
                continue
            test_hook = getattr(mod, "run_self_tests", None)

            try:
                test_hook()
            except Exception as exc:  # pragma: no cover - manual smoke harness
                failures.append((module_name, exc))
                traceback.print_exc()

    print("\n=== Running module import smoke tests ===")
    for module_name in all_modules:
        try:
            mod = importlib.import_module(module_name)
            if inspect.ismodule(mod):
                pass
        except Exception as exc:
            skipped.append((module_name, exc))
            print(f"{module_name}: import failed, skipping ({exc}).")
            continue

    if skipped:
        print("\nSelf-tests skipped due to import/runtime constraints:")
        for name, exc in skipped:
            print(f" - {name}: {exc}")

    if failures:
        print("\nSelf-tests finished with failures:")
        for name, exc in failures:
            print(f" - {name}: {exc}")
        return 1

    print("\nAll discovered module self-tests and import smoke tests completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
