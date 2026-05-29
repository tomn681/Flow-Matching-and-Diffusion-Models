"""Unified command-line entrypoint for training and inference workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import train as train_entry
from src import run_model as run_model_entry


_RUN_MODEL_MODES = {
    "sample",
    "encode",
    "decode",
    "evaluate",
    "build_tensor_cache",
    "debug_compare",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m genlib",
        description="Unified genlib CLI for training and model runtime tasks.",
    )
    parser.add_argument(
        "command",
        choices=("train", "sample", "encode", "decode", "evaluate", "build_tensor_cache", "debug_compare"),
        help="Top-level command to execute.",
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

    raise ValueError(f"Unsupported command '{ns.command}'.")
