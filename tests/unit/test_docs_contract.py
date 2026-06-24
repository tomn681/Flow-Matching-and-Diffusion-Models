from __future__ import annotations

import importlib
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_mkdocs_nav_markdown_targets_exist() -> None:
    mkdocs = (_repo_root() / "mkdocs.yml").read_text(encoding="utf-8")
    rel_paths = re.findall(r":\s*([\w./-]+\.md)\s*$", mkdocs, flags=re.MULTILINE)
    assert rel_paths, "No markdown targets found in mkdocs.yml nav."

    missing = [p for p in rel_paths if not (_repo_root() / "docs" / p).exists()]
    assert not missing, f"Missing nav pages: {missing}"


def test_api_reference_modules_are_importable() -> None:
    api_dir = _repo_root() / "docs" / "api"
    missing_targets: list[str] = []
    bad_imports: list[str] = []

    for page in sorted(api_dir.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        matches = re.findall(r"^:::\s*([\w.]+)\s*$", text, flags=re.MULTILINE)
        if not matches:
            missing_targets.append(page.name)
            continue
        for module_name in matches:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - reported with full detail
                bad_imports.append(f"{page.name}: {module_name} ({exc})")

    assert not missing_targets, f"API pages without mkdocstrings target: {missing_targets}"
    assert not bad_imports, f"Unimportable API modules: {bad_imports}"


def test_core_getting_started_and_readme_config_paths_exist() -> None:
    root = _repo_root()
    pages = [
        root / "README.md",
        root / "docs" / "guides" / "getting_started.md",
        root / "docs" / "tutorials" / "getting-started.md",
        root / "docs" / "guides" / "latent_diffusion.md",
    ]
    config_refs = [
        "configs/autoencoder_kl_small.json",
        "configs/LDCT/vae/vae_exp_r0_recon_only.json",
    ]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for rel in config_refs:
            if rel in text:
                assert (root / rel).exists(), f"{page} references missing config {rel}"
