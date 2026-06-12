from __future__ import annotations

import tomllib
from pathlib import Path


def test_py_typed_marker_exists() -> None:
    marker = Path(__file__).resolve().parents[2] / "src" / "py.typed"
    assert marker.exists()


def test_mypy_config_exists() -> None:
    cfg = Path(__file__).resolve().parents[2] / "mypy.ini"
    assert cfg.exists()
    text = cfg.read_text(encoding="utf-8")
    assert "[mypy]" in text


def test_public_api_typecheck_script_exists() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "typecheck_public_api.sh"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "--follow-imports=silent" in text
    assert "genlib/__init__.py" in text


def test_public_package_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]

    import genlib

    assert genlib.__version__ == expected
