from __future__ import annotations

from pathlib import Path

import sampling.generative_sampler as generative_sampler_mod
import sampling.vae_sampler as vae_sampler_mod
from sampling.generative_sampler import DiffusionSampler
from sampling.vae_sampler import VAESampler


def test_vae_sampler_method_delegation(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def _capture(name):
        def _fn(**kwargs):
            calls.append((name, kwargs))
        return _fn

    monkeypatch.setattr(vae_sampler_mod.autoencoder_sampler, "encode", _capture("encode"))
    monkeypatch.setattr(vae_sampler_mod.autoencoder_sampler, "decode", _capture("decode"))
    monkeypatch.setattr(vae_sampler_mod.autoencoder_sampler, "sample", _capture("sample"))
    monkeypatch.setattr(vae_sampler_mod.autoencoder_sampler, "evaluate", _capture("evaluate"))
    monkeypatch.setattr(vae_sampler_mod.autoencoder_sampler, "debug_compare", _capture("debug_compare"))

    sampler = VAESampler(
        ckpt_dir=tmp_path,
        save=True,
        batch_size=4,
        device="cpu",
        seed=11,
        timestep=5,
        num_samples=3,
        save_input=True,
        save_conditioning=True,
        save_diff_map=True,
        diff_amplify=7.5,
        save_tensor_cache=True,
        disable_tensor_cache=True,
        use_ema=True,
    )
    sampler.encode()
    sampler.decode()
    sampler.sample()
    sampler.evaluate()
    sampler.debug_compare()

    names = [n for n, _ in calls]
    assert names == ["encode", "decode", "sample", "evaluate", "debug_compare"]
    encode_kwargs = calls[0][1]
    assert encode_kwargs["timestep"] == 5
    assert encode_kwargs["save_tensor_cache"] is True
    assert encode_kwargs["disable_tensor_cache"] is True
    assert encode_kwargs["use_ema"] is True
    assert calls[2][1]["save_diff_map"] is True
    assert calls[2][1]["diff_amplify"] == 7.5


def test_base_sampler_rejects_non_positive_diff_amplify(tmp_path: Path) -> None:
    try:
        VAESampler(ckpt_dir=tmp_path, diff_amplify=0.0)
        raise AssertionError("Expected ValueError for non-positive diff_amplify.")
    except ValueError as exc:
        assert "diff_amplify must be positive" in str(exc)


def test_generative_sampler_method_delegation(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def _capture(name):
        def _fn(**kwargs):
            calls.append((name, kwargs))
        return _fn

    monkeypatch.setattr(generative_sampler_mod, "_run_encode", _capture("encode"))
    monkeypatch.setattr(generative_sampler_mod, "_run_decode", _capture("decode"))
    monkeypatch.setattr(generative_sampler_mod, "_run_evaluate", _capture("evaluate"))
    monkeypatch.setattr(generative_sampler_mod, "_run_debug_compare", _capture("debug_compare"))

    sampler = DiffusionSampler(
        ckpt_dir=tmp_path,
        save=True,
        batch_size=2,
        device="cpu",
        seed=99,
        timestep=8,
        num_samples=4,
        save_input=True,
        save_conditioning=True,
        num_inference_steps=12,
        start_step=10,
        last_n_steps=3,
        scheduler="ddpm",
        save_tensor_cache=True,
        disable_tensor_cache=True,
    )
    sampler.encode()
    sampler.decode()
    sampler.sample()
    sampler.evaluate()
    sampler.debug_compare()

    names = [n for n, _ in calls]
    assert names == ["encode", "decode", "decode", "evaluate", "debug_compare"]
    for _, kwargs in calls:
        assert kwargs["model_type"] == "diffusion"
    assert calls[0][1]["timestep"] == 8
    assert calls[0][1]["disable_tensor_cache"] is True
    assert calls[1][1]["num_inference_steps"] == 12
    assert calls[4][1]["scheduler"] == "ddpm"
