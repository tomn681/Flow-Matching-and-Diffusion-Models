from __future__ import annotations

from pathlib import Path


def test_mkdocs_scaffold_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    mkdocs = root / "mkdocs.yml"
    assert mkdocs.exists()

    required_pages = [
        root / "docs/index.md",
        root / "docs/tutorials/getting-started.md",
        root / "docs/tutorials/training-workflows.md",
        root / "docs/tutorials/inference-workflows.md",
        root / "docs/architecture/overview.md",
        root / "docs/architecture/registries-and-plugins.md",
        root / "docs/api/core.md",
        root / "docs/api/protocols.md",
        root / "docs/api/registries.md",
        root / "docs/api/configs.md",
        root / "docs/api/models.md",
        root / "docs/api/training.md",
        root / "docs/api/pipelines.md",
        root / "docs/api/sampling.md",
        root / "docs/api/scheduling.md",
        root / "docs/api/noise.md",
        root / "docs/api/losses.md",
    ]
    missing = [str(p) for p in required_pages if not p.exists()]
    assert not missing, f"Missing documentation pages: {missing}"
