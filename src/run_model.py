"""
Unified dispatcher for sampling/encoding/decoding/evaluation workflows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from core.protocols import Decodable, Encodable, Evaluatable, Reflowable, Sampleable
from sampling.base import BaseSampler
from sampling import SAMPLER_REGISTRY
from utils.sampling_utils import load_run_config


def _resolve_sampler(model_type: str):
    key = str(model_type).lower()
    return SAMPLER_REGISTRY.get(key)


def _supports_mode(sampler, mode: str) -> bool:
    mode_key = str(mode).strip().lower()
    sampler_type = type(sampler)

    if mode_key == "encode":
        return isinstance(sampler, Encodable) and sampler_type.encode is not BaseSampler.encode
    if mode_key == "decode":
        return isinstance(sampler, Decodable) and sampler_type.decode is not BaseSampler.decode
    if mode_key == "sample":
        return (
            isinstance(sampler, Sampleable)
            and (
                sampler_type.sample is not BaseSampler.sample
                or sampler_type.decode is not BaseSampler.decode
            )
        )
    if mode_key == "evaluate":
        return isinstance(sampler, Evaluatable) and sampler_type.evaluate is not BaseSampler.evaluate
    if mode_key == "generate_reflow_pairs":
        return isinstance(sampler, Reflowable)
    if mode_key in {"build_tensor_cache", "debug_compare"}:
        return True
    return False


def main(argv: list[str] | None = None) -> None:
    """
    Dispatch a model workflow from a checkpoint directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    parser = argparse.ArgumentParser(description="Run sampling/encoding/decoding/eval/cache-build from a checkpoint dir.")
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="Checkpoint directory containing train_config.json.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("sample", "encode", "decode", "evaluate", "build_tensor_cache", "debug_compare", "generate_reflow_pairs"),
        default="sample",
    )
    parser.add_argument("--data_txt", type=str, default=None, help="Optional override split file.")
    parser.add_argument("--save", action="store_true", help="Save outputs to disk.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory (defaults to ckpt_dir/outputs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timestep", type=int, default=None, help="Optional timestep for encode.")
    parser.add_argument("--num_samples", type=int, default=None, help="Random subset size to process.")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Override scheduler inference steps (diffusion/flow only).")
    parser.add_argument("--start_step", type=int, default=None, help="Start denoising from this train-timestep index (e.g., 700 runs from t<=700).")
    parser.add_argument("--last_n_steps", type=int, default=None, help="Run only the last N denoising steps.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Override scheduler at runtime (ddpm, ddim, dpmsolver1, dpmsolver2, dpmsolver++, dpmsolversde, unipc, flowmatch).",
    )
    parser.add_argument("--save_input", action="store_true", help="Also save model inputs when --save is enabled.")
    parser.add_argument("--save_conditioning", action="store_true", help="Also save conditioning tensors when --save is enabled.")
    parser.add_argument(
        "--save_tensor_cache",
        action="store_true",
        help="Force writing tensor cache files at runtime without editing train_config.json.",
    )
    parser.add_argument("--num_pairs", type=int, default=None, help="Number of reflow pairs to generate in generate_reflow_pairs mode.")
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    cfg = load_run_config(args.ckpt_dir)
    model_type = cfg.get("model", {}).get("model_type", "vae")
    if str(model_type).lower() in {"latent_diffusion", "latent_flow_matching", "latent_rectified_flow"}:
        supported = {"sample", "decode"}
        if args.mode not in supported:
            allowed = ", ".join(sorted(supported))
            raise ValueError(f"Mode '{args.mode}' is not supported for '{model_type}'. Supported modes: {allowed}.")
    sampler_cls = _resolve_sampler(model_type)

    sampler = sampler_cls(
        ckpt_dir=args.ckpt_dir,
        data_txt=args.data_txt,
        save=args.save,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        timestep=args.timestep,
        num_samples=args.num_samples,
        save_input=args.save_input,
        save_conditioning=args.save_conditioning,
        num_inference_steps=args.num_inference_steps,
        start_step=args.start_step,
        last_n_steps=args.last_n_steps,
        scheduler=args.scheduler,
        save_tensor_cache=args.save_tensor_cache,
        num_pairs=args.num_pairs,
    )

    with torch.no_grad():
        method = getattr(sampler, args.mode, None)
        if method is None:
            raise ValueError(f"Unknown mode '{args.mode}'.")
        if not _supports_mode(sampler, args.mode):
            raise ValueError(
                f"Mode '{args.mode}' is not implemented by sampler '{type(sampler).__name__}'."
            )
        method()


if __name__ == "__main__":
    main()
