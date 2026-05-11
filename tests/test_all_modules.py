from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _all_modules() -> list[str]:
    root = Path(__file__).resolve().parents[1] / "src"
    modules = []
    for py in root.rglob("*.py"):
        rel = py.relative_to(root.parent).with_suffix("")
        modules.append(".".join(rel.parts))
    return sorted(modules)


@pytest.mark.parametrize("module_name", _all_modules())
def test_module_import_smoke(module_name: str):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Optional heavy dependencies may be unavailable in minimal environments.
        if exc.name in {"torch", "torchvision", "diffusers", "pydicom", "skimage", "pandas", "numpy", "PIL"}:
            pytest.skip(f"Optional dependency missing for {module_name}: {exc.name}")
        raise


@pytest.mark.parametrize("module_name", _all_modules())
def test_run_self_tests_hook_if_present(module_name: str):
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "torchvision", "diffusers", "pydicom", "skimage", "pandas", "numpy", "PIL"}:
            pytest.skip(f"Optional dependency missing for {module_name}: {exc.name}")
        raise

    hook = getattr(mod, "run_self_tests", None)
    if hook is None:
        return
    hook()
