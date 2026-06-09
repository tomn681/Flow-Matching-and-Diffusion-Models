"""
Trainer-facing composable loss components and the loss registry.

Boundary:
- `src.losses` owns components, composition, and registration.
- `src.nn.losses` owns the tensor-level math and small discriminator
  primitives these components are built on.
"""

import sys as _sys

if __name__ == "src.losses" and "losses" in _sys.modules:
    _canonical = _sys.modules["losses"]
    _sys.modules[__name__] = _canonical
    globals().update(_canonical.__dict__)
else:
    from .assembler import LossAssembler
    from .base import BaseLossComponent
    from .adversarial import GANDiscriminatorLoss, GANGeneratorLoss
    from .gradient import GradientLoss
    from .perceptual import PerceptualLossComponent
    from .regularization import KLLoss, VQLoss
    from .registry import LOSS_REGISTRY
    from .ssim import SSIMLoss

    # Import reconstruction module for registry side effects.
    from . import gradient as _gradient  # noqa: F401
    from . import reconstruction as _reconstruction  # noqa: F401
    from . import ssim as _ssim  # noqa: F401

    __all__ = [
        "BaseLossComponent",
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
    _alt_prefix = "losses." if __name__ == "src.losses" else "src.losses."
    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod_name.startswith(_prefix):
            _alias = _alt_prefix + _mod_name[len(_prefix):]
            _sys.modules.setdefault(_alias, _mod)
    del _prefix, _alt_prefix, _mod_name, _mod

del _sys
