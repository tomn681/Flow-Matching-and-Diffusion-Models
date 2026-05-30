from __future__ import annotations

from utils import sampling_utils


def test_build_tensor_cache_from_config_prefers_dataset_build_cache(monkeypatch) -> None:
    class _Dataset:
        def build_cache(self):
            return 13

    monkeypatch.setattr(sampling_utils, "build_sampling_dataset", lambda *_args, **_kwargs: _Dataset())
    total = sampling_utils.build_tensor_cache_from_config(
        cfg={"training": {}},
        data_txt=None,
        batch_size=4,
        seed=42,
        num_samples=10,
    )
    assert total == 13

