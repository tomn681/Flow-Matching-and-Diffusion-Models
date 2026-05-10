"""
Unified dispatcher for sampling/encoding/decoding/evaluation workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.sampling_utils import load_run_config
from pipelines.samplers.handlers import DiffusionHandler, FlowMatchingHandler, VAEHandler


HANDLER_REGISTRY = {
    "vae": VAEHandler,
    "diffusion": DiffusionHandler,
    "flow_matching": FlowMatchingHandler,
}


def _resolve_handler(model_type: str):
    key = str(model_type).lower()
    if key not in HANDLER_REGISTRY:
        raise ValueError(f"Unsupported model_type '{model_type}'.")
    return HANDLER_REGISTRY[key]


def main() -> None:
    """
    Dispatch a model workflow from a checkpoint directory.
    """
    parser = argparse.ArgumentParser(description="Run sampling/encoding/decoding/eval from a checkpoint dir.")
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="Checkpoint directory containing train_config.json.")
    parser.add_argument("--mode", type=str, choices=("sample", "encode", "decode", "evaluate"), default="sample")
    parser.add_argument("--data_txt", type=str, default=None, help="Optional override split file.")
    parser.add_argument("--save", action="store_true", help="Save outputs to disk.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory (defaults to ckpt_dir/outputs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timestep", type=int, default=None, help="Optional timestep for encode.")
    parser.add_argument("--num_samples", type=int, default=None, help="Random subset size to process.")
    parser.add_argument("--save_input", action="store_true", help="Also save model inputs when --save is enabled.")
    parser.add_argument("--save_conditioning", action="store_true", help="Also save conditioning tensors when --save is enabled.")
    args = parser.parse_args()

    cfg = load_run_config(args.ckpt_dir)
    model_type = cfg.get("model", {}).get("model_type", "vae")
    handler_cls = _resolve_handler(model_type)

    handler = handler_cls(
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
    )

    if args.mode == "encode":
        handler.encode()
    elif args.mode == "decode":
        handler.decode()
    elif args.mode == "evaluate":
        handler.evaluate()
    else:
        handler.sample()


if __name__ == "__main__":
    main()
