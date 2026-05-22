from __future__ import annotations

import inspect

from models.vae.base import BaseVAE


def test_base_vae_forward_has_sample_posterior_default_true() -> None:
    sig = inspect.signature(BaseVAE.forward)
    assert "sample_posterior" in sig.parameters
    param = sig.parameters["sample_posterior"]
    assert param.default is True

