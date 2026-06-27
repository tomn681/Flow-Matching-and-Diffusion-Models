from __future__ import annotations

import pytest
import torch

from src.utils.model_utils.diffusion_loading import _load_legacy_unet_state


class _FakeModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = dict(state)
        self.loaded = None
        self.loaded_strict = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded = dict(state_dict)
        self.loaded_strict = bool(strict)
        return None


def test_load_legacy_unet_state_uses_validated_mapping_and_strict_load() -> None:
    tensor = torch.randn(4, 4, 3, 3)
    model = _FakeModel({"block.conv1.conv.weight": torch.empty_like(tensor)})

    _load_legacy_unet_state(model, {"block.conv1.weight": tensor}, strict_shapes=True)

    assert model.loaded_strict is True
    assert model.loaded is not None
    assert "block.conv1.conv.weight" in model.loaded
    assert torch.equal(model.loaded["block.conv1.conv.weight"], tensor)


def test_load_legacy_unet_state_raises_clear_error_on_unvalidated_mapping_failure() -> None:
    tensor = torch.randn(4, 4, 3, 3)
    model = _FakeModel({"other.weight": torch.empty_like(tensor)})

    with pytest.raises(RuntimeError, match="Use model\\.unet\\.unet_impl='hf_diffusers'"):
        _load_legacy_unet_state(model, {"block.conv1.weight": tensor}, strict_shapes=True)


def test_load_legacy_unet_state_allows_non_strict_fallback_when_requested() -> None:
    tensor = torch.randn(4, 4, 3, 3)
    model = _FakeModel({"other.weight": torch.empty_like(tensor)})

    _load_legacy_unet_state(model, {"block.conv1.weight": tensor}, strict_shapes=False)

    assert model.loaded_strict is False
    assert model.loaded is not None
    assert "block.conv1.conv.weight" in model.loaded
