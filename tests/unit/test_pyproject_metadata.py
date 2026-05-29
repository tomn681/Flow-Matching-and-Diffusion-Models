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
    assert entry_points["genlib.plugins"]["builtin"] == "src.plugins.builtin:register"
