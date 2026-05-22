from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


FORBIDDEN_IMPORTS = {
    "core": {"models", "training", "sampling", "pipelines", "datasets", "losses", "noise", "scheduling", "utils", "compat", "configs", "nn"},
    "nn": {"models", "training", "sampling", "pipelines", "datasets", "compat"},
    "models": {"training", "sampling", "pipelines", "compat"},
    "noise": {"training", "sampling", "pipelines", "compat", "models"},
    "losses": {"training", "sampling", "pipelines", "compat"},
    "scheduling": {"training", "sampling", "pipelines", "compat"},
    "configs": {"training", "sampling", "pipelines", "compat", "models", "noise", "losses", "scheduling", "datasets", "nn", "utils"},
}


def _iter_pkg_files(pkg: str):
    pkg_dir = SRC_ROOT / pkg
    yield from pkg_dir.rglob("*.py")


def _bad_imports_for_file(py_file: Path, forbidden: set[str]) -> list[tuple[int, str]]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in forbidden:
                    bad.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative imports are package-local by construction.
                continue
            if node.module:
                top = node.module.split(".")[0]
                if top in forbidden:
                    bad.append((node.lineno, node.module))
    return bad


def test_import_boundaries() -> None:
    violations: list[str] = []
    for pkg, forbidden in FORBIDDEN_IMPORTS.items():
        for py_file in _iter_pkg_files(pkg):
            for line, mod in _bad_imports_for_file(py_file, forbidden):
                violations.append(f"{py_file}:{line} imports forbidden module '{mod}' for package '{pkg}'")
    assert not violations, "Import boundary violations:\n" + "\n".join(violations)

