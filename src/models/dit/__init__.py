import sys as _sys

_canonical_name = None
if __name__ in {"src.models.dit", "genlib.models.dit"} and "models.dit" in _sys.modules:
    _canonical_name = "models.dit"
elif __name__ == "models.dit" and "src.models.dit" in _sys.modules:
    _canonical_name = "src.models.dit"
elif __name__ == "models.dit" and "genlib.models.dit" in _sys.modules:
    _canonical_name = "genlib.models.dit"

if _canonical_name is not None:
    _canonical = _sys.modules[_canonical_name]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from . import dit
    from .dit import DiTND

    __all__ = ["DiTND", "dit"]

    _prefix = f"{__name__}."
    for _mod_name, _mod in list(_sys.modules.items()):
        if not _mod_name.startswith(_prefix):
            continue
        _suffix = _mod_name[len(_prefix):]
        for _alias_prefix in ("models.dit.", "src.models.dit.", "genlib.models.dit."):
            _sys.modules.setdefault(_alias_prefix + _suffix, _mod)

del _sys
