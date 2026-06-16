# Changelog

## 1.0.0

Stable public release after Wave 2 closure of the Phase 1–6 audit program.

### Added
- Reflow pair generation and reflow training/sampling support
- LoRA wrapping, save/load, and runtime integration
- Supervised `UNetTrainer`
- Distillation trainer and sampler support
- DiT model integration
- Multi-resolution training schedule support
- Temporal attention, video dataset, video UNet, and medical 3D dataset support
- Multimodal conditioning with text adapter support
- Model merging utility
- GAN training enhancements
- Public API reference, tutorial guides, and CLI help coverage
- Public API typecheck script

### Changed
- Migrated `SCHEDULER_REGISTRY` to `Registry[T]`
- Expanded validation script to cover post-K features
- Added `__version__` to the public package API
- Declared typed-package support via `py.typed`

### Fixed
- Reflow runtime path and checkpoint resolution in sampler-driven generation
- Remaining framework-side capability checks now use protocols instead of `hasattr`
- Version metadata now aligned between `src/__init__.py` and `pyproject.toml`
- Versioned safetensors+sidecar checkpoint contract with explicit format metadata
- Canonical `genlib` package root now coexists safely with `src` and flat compatibility imports

## 0.9.0

Temporary downgrade from `1.0.0` while the Phase 1–6 audit blockers remained open.

### Notes
- Public API surface passes the project-owned `mypy` check via `scripts/typecheck_public_api.sh`
- Full test suite baseline at release preparation: `545 passed`
