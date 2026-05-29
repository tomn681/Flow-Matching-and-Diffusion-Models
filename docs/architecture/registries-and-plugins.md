# Registries and Plugins

## Registries

The project defines registries for:

- models
- losses
- noise processes
- schedulers
- samplers
- trainers

Registries are the core extension point for adding concrete implementations
without changing dispatch code.

## Plugin Discovery

Plugin discovery uses Python entry points through group `genlib.plugins`.
The discovery helper is `core.plugin.discover_plugins()`.

`pyproject.toml` defines the canonical plugin entry-point group for packaging.
