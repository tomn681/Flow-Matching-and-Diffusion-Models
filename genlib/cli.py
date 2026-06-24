"""Unified command-line entrypoint for training and inference workflows."""

from __future__ import annotations

import argparse
import importlib

import torch

class _LazyEntryModule:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def main(self, argv=None):
        return self._load().main(argv)


run_model_entry = _LazyEntryModule("genlib.run_model")
train_entry = _LazyEntryModule("genlib.train")


def generate_reflow_pairs(**kwargs):
    fn = importlib.import_module("genlib.noise.reflow").generate_reflow_pairs
    return fn(**kwargs)


def build_scheduler(*args, **kwargs):
    fn = importlib.import_module("genlib.scheduling.builder").build_scheduler
    return fn(*args, **kwargs)


def load_json_config(*args, **kwargs):
    fn = importlib.import_module("genlib.utils").load_json_config
    return fn(*args, **kwargs)


def build_diffusion_model(*args, **kwargs):
    fn = importlib.import_module("genlib.utils.model_utils.diffusion_loading").build_diffusion_model
    return fn(*args, **kwargs)


_RUN_MODEL_MODES = {
    "sample",
    "encode",
    "decode",
    "evaluate",
    "build_tensor_cache",
    "debug_compare",
}


def _parse_shape(shape_text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in shape_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("sample_shape must contain at least one dimension.")
    shape = tuple(int(p) for p in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError("sample_shape dimensions must be > 0.")
    return shape


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m genlib",
        description="Unified genlib CLI for training and model runtime tasks.",
        epilog=(
            "Examples:\n"
            "  python -m genlib train --config configs/LDCT/vae/vae_sd_kl_bce_focal_ldct.json\n"
            "  python -m genlib sample --ckpt_dir checkpoints/run1 --save\n"
            "  python -m genlib evaluate --ckpt_dir checkpoints/run1 --batch_size 8\n"
            "  python -m genlib generate-reflow-pairs --config cfg.json --ckpt model.pt --num-pairs 50000 --output-dir ./pairs --sample-shape 1,32,32"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=("train", "sample", "encode", "decode", "evaluate", "build_tensor_cache", "debug_compare", "generate-reflow-pairs"),
        help=(
            "Top-level command to execute:\n"
            "  train: forward to packaged training entrypoint\n"
            "  sample|encode|decode|evaluate|build_tensor_cache|debug_compare: forward to run_model.py\n"
            "  generate-reflow-pairs: direct utility for reflow pair export"
        ),
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected command.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    forwarded = list(ns.args)

    if ns.command == "train":
        train_entry.main(forwarded)
        return

    if ns.command in _RUN_MODEL_MODES:
        run_model_entry.main(["--mode", ns.command, *forwarded])
        return

    if ns.command == "generate-reflow-pairs":
        pair_parser = argparse.ArgumentParser(
            prog="python -m genlib generate-reflow-pairs",
            description="Generate (z0, z1) coupling pairs for reflow training.",
            epilog=(
                "Example:\n"
                "  python -m genlib generate-reflow-pairs --config cfg.json --ckpt model.pt "
                "--num-pairs 50000 --output-dir ./pairs --sample-shape 1,32,32 --batch-size 8"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        pair_parser.add_argument("--config", type=str, required=True, help="Path to training config JSON.")
        pair_parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for trained flow model.")
        pair_parser.add_argument("--num-pairs", type=int, required=True, help="Number of pairs to generate.")
        pair_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for .pt pair files.")
        pair_parser.add_argument("--sample-shape", type=str, required=True, help="Comma-separated sample shape, e.g. '4,32,32'.")
        pair_parser.add_argument("--num-inference-steps", type=int, default=None, help="Override number of scheduler inference steps.")
        pair_parser.add_argument("--batch-size", type=int, default=4, help="Pair generation batch size.")
        pair_parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda.")
        pair_args = pair_parser.parse_args(forwarded)

        cfg = load_json_config(pair_args.config)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(pair_args.device or default_device)
        model = build_diffusion_model(cfg, device, ckpt_path=pair_args.ckpt, set_eval=True)
        scheduler_cfg = cfg.get("model", {}).get("scheduler", {})
        training_cfg = cfg.get("training", {})
        scheduler, steps = build_scheduler(scheduler_cfg, training_cfg)
        sample_shape = _parse_shape(pair_args.sample_shape)
        if len(sample_shape) < 2:
            raise ValueError("sample_shape must include channel and spatial dims, e.g. '4,32,32'.")
        if pair_args.num_inference_steps is not None:
            steps = int(pair_args.num_inference_steps)

        generate_reflow_pairs(
            model=model,
            scheduler=scheduler,
            num_pairs=int(pair_args.num_pairs),
            sample_shape=sample_shape,
            device=device,
            output_dir=pair_args.output_dir,
            num_inference_steps=steps,
            batch_size=int(pair_args.batch_size),
        )
        return

    raise ValueError(f"Unsupported command '{ns.command}'.")
