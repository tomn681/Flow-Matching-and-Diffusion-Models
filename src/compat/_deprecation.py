from __future__ import annotations

import warnings


DEFAULT_REMOVAL_VERSION = "2.0"


def warn_deprecated(
    *,
    api: str,
    replacement: str,
    remove_in: str = DEFAULT_REMOVAL_VERSION,
    stacklevel: int = 2,
) -> None:
    warnings.warn(
        f"`{api}` is deprecated and will be removed in v{remove_in}. "
        f"Use `{replacement}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


__all__ = ["DEFAULT_REMOVAL_VERSION", "warn_deprecated"]

