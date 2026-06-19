from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_declares_genlib_cli_and_plugin_group() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    project = data["project"]
    scripts = project["scripts"]
    entry_points = project["entry-points"]

    assert scripts["genlib"] == "genlib.cli:main"
    assert "genlib.plugins" in entry_points
    assert entry_points["genlib.plugins"]["builtin"] == "genlib.plugins.builtin:register"


def test_pyproject_declares_optional_runtime_extras() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    assert "text" in extras
    assert "medical" in extras
    assert "perceptual" in extras
    assert "tracking" in extras
    assert any(dep.startswith("transformers") for dep in extras["text"])
    assert any(dep.startswith("pydicom") for dep in extras["medical"])
    assert any(dep.startswith("lpips") for dep in extras["perceptual"])
    assert any(dep.startswith("wandb") for dep in extras["tracking"])
