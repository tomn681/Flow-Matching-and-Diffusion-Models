"""
Trainer-facing composable loss components and the loss registry.

Boundary:
- `src.losses` owns components, composition, and registration.
- `src.nn.losses` owns the tensor-level math and small discriminator
  primitives these components are built on.
"""

import sys as _sys

_canonical_name = None
if __name__ in {"src.losses", "genlib.losses"} and "losses" in _sys.modules:
    _canonical_name = "losses"
elif __name__ == "losses" and "src.losses" in _sys.modules:
    _canonical_name = "src.losses"
elif __name__ == "losses" and "genlib.losses" in _sys.modules:
    _canonical_name = "genlib.losses"

if _canonical_name is not None:
    _canonical = _sys.modules[_canonical_name]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .assembler import LossAssembler
    from .base import BaseLossComponent
    from .adversarial import GANDiscriminatorLoss, GANGeneratorLoss
    from .denoising import DenoisingMSELoss
    from .gradient import GradientLoss
    from .perceptual import PerceptualLossComponent
    from .regularization import KLLoss, VQLoss
    from .registry import LOSS_REGISTRY
    from .ssim import SSIMLoss

    # Import reconstruction module for registry side effects.
    from . import adversarial as _adversarial  # noqa: F401
    from . import denoising as _denoising  # noqa: F401
    from . import gradient as _gradient  # noqa: F401
    from . import perceptual as _perceptual  # noqa: F401
    from . import reconstruction as _reconstruction  # noqa: F401
    from . import regularization as _regularization  # noqa: F401
    from . import ssim as _ssim  # noqa: F401

    __all__ = [
        "BaseLossComponent",
        "DenoisingMSELoss",
        "GANDiscriminatorLoss",
        "GANGeneratorLoss",
        "GradientLoss",
        "KLLoss",
        "LossAssembler",
        "LOSS_REGISTRY",
        "PerceptualLossComponent",
        "VQLoss",
        "SSIMLoss",
    ]

    _prefix = f"{__name__}."
    for _mod_name, _mod in list(_sys.modules.items()):
        if not _mod_name.startswith(_prefix):
            continue
        _suffix = _mod_name[len(_prefix):]
        for _alias_prefix in ("losses.", "src.losses.", "genlib.losses."):
            _sys.modules.setdefault(_alias_prefix + _suffix, _mod)

del _sys
