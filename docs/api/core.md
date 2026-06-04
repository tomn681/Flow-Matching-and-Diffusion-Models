# Core API

Core types, registry primitives, and protocol contracts used across the
framework.

## Common Types

These are the shared dataclasses and helpers used by trainers, samplers, and
model wrappers.

::: src.core.types

## Registry Primitive

All first-class registries are built on the same generic `Registry[T]`
implementation.

```python
from src.core.registry import Registry

registry = Registry("example")

@registry.register("toy")
class Toy:
    pass
```

::: src.core.registry

## Protocols

The protocol layer defines framework contracts and capabilities without forcing
inheritance.

::: src.core.protocols
