"""
Sampling, encoding, decoding, and evaluation for flow-matching models.
"""

from __future__ import annotations

from pathlib import Path

from .diffusion_like import _run_decode, _run_encode, _run_evaluate


def encode(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    timestep: int | None = None,
    num_samples: int | None = None,
) -> None:
    _run_encode(
        ckpt_dir=ckpt_dir,
        model_type="flow_matching",
        data_txt=data_txt,
        save=save,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
        timestep=timestep,
        num_samples=num_samples,
    )


def decode(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
) -> None:
    _run_decode(
        ckpt_dir=ckpt_dir,
        model_type="flow_matching",
        data_txt=data_txt,
        save=save,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
        num_samples=num_samples,
        save_input=save_input,
        save_conditioning=save_conditioning,
    )


def sample(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
) -> None:
    decode(
        ckpt_dir=ckpt_dir,
        data_txt=data_txt,
        save=save,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
        num_samples=num_samples,
        save_input=save_input,
        save_conditioning=save_conditioning,
    )


def evaluate(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
) -> None:
    _run_evaluate(
        ckpt_dir=ckpt_dir,
        model_type="flow_matching",
        data_txt=data_txt,
        save=save,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
        num_samples=num_samples,
        save_input=save_input,
        save_conditioning=save_conditioning,
    )
