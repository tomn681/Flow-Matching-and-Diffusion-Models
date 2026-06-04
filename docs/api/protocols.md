# Protocols API

Implementation contracts and capability protocols used throughout the library.

## How to Read These Protocols

- Contract protocols describe the minimum interface expected by framework code.
- Capability protocols describe optional features that only some components
  support.
- Prefer implementing a protocol directly over inheriting from a placeholder
  base class when the behavior is optional.

## Examples

`Sampleable` / `Evaluatable` style capability:

```python
from typing import Protocol

class Sampleable(Protocol):
    def sample(self) -> None: ...
```

`ResolutionSchedule` style runtime contract:

```python
class MySchedule:
    stages = []

    def current_resolution(self, epoch: int) -> int:
        return 128
```

::: src.core.protocols
