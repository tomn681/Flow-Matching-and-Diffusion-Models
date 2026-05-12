"""
Shared helpers for sampling/encoding/decoding dispatchers.
"""

from __future__ import annotations

import random
import csv
import json
from pathlib import Path

from utils import build_dataset_from_config, load_json_config
from utils.dataset_utils import iter_batches


def _load_diffusers_legacy_run_config(ckpt_dir: Path) -> dict:
    """
    Build a minimal run config from a legacy Diffusers pipeline folder:
      - model_index.json
      - scheduler/scheduler_config.json
      - unet/config.json (or config.txt)
    """
    model_index_path = ckpt_dir / "model_index.json"
    scheduler_cfg_path = ckpt_dir / "scheduler" / "scheduler_config.json"
    unet_cfg_path_json = ckpt_dir / "unet" / "config.json"
    unet_cfg_path_txt = ckpt_dir / "unet" / "config.txt"
    unet_cfg_path = unet_cfg_path_json if unet_cfg_path_json.exists() else unet_cfg_path_txt

    if not (model_index_path.exists() and scheduler_cfg_path.exists() and unet_cfg_path.exists()):
        raise FileNotFoundError(
            "Missing train_config.json and could not resolve a legacy diffusers folder layout."
        )

    model_index = json.loads(model_index_path.read_text())
    scheduler_cfg = json.loads(scheduler_cfg_path.read_text())
    unet_cfg = json.loads(unet_cfg_path.read_text())

    in_channels = int(unet_cfg.get("in_channels", 1))
    out_channels = int(unet_cfg.get("out_channels", 1))
    channels = out_channels
    conditioning = "concatenate" if in_channels > out_channels else None

    cfg = {
        "training": {
            "data_root": "/",
            "dataset": "ldct",
            "channels": channels,
            "img_size": int(unet_cfg.get("sample_size", 256)),
            "num_train_timesteps": int(scheduler_cfg.get("num_train_timesteps", 1000)),
            "num_inference_steps": int(scheduler_cfg.get("num_train_timesteps", 1000)),
            "conditioning": conditioning,
            "load_ldct": bool(conditioning in {"concatenate", "attention"}),
            "norm": True,
        },
        "model": {
            "model_type": "diffusion",
            "conditioning": conditioning,
            "scheduler": {
                "name": str(scheduler_cfg.get("_class_name", "DDPMScheduler")).replace("Scheduler", "").lower(),
                "num_train_timesteps": int(scheduler_cfg.get("num_train_timesteps", 1000)),
                "num_inference_steps": int(scheduler_cfg.get("num_train_timesteps", 1000)),
                "params": {
                    k: v
                    for k, v in scheduler_cfg.items()
                    if k
                    not in {
                        "_class_name",
                        "_diffusers_version",
                        "num_train_timesteps",
                        "num_inference_steps",
                        "trained_betas",
                    }
                },
            },
            "unet": {
                "unet_impl": "diffusers_nd",
                "in_channels_already_conditioned": True,
                "sample_size": unet_cfg.get("sample_size", 256),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "layers_per_block": int(unet_cfg.get("layers_per_block", 2)),
                "block_out_channels": tuple(unet_cfg.get("block_out_channels", [128, 128, 256, 256, 512, 512])),
                "down_block_types": tuple(unet_cfg.get("down_block_types", [])),
                "up_block_types": tuple(unet_cfg.get("up_block_types", [])),
                "attention_head_dim": int(unet_cfg.get("attention_head_dim", 8)),
                "norm_num_groups": int(unet_cfg.get("norm_num_groups", 32)),
                "norm_eps": float(unet_cfg.get("norm_eps", 1e-5)),
                "flip_sin_to_cos": bool(unet_cfg.get("flip_sin_to_cos", True)),
                "freq_shift": int(unet_cfg.get("freq_shift", 0)),
                "center_input_sample": bool(unet_cfg.get("center_input_sample", False)),
                "resnet_time_scale_shift": str(unet_cfg.get("resnet_time_scale_shift", "default")),
                "add_attention": bool(unet_cfg.get("add_attention", True)),
            },
            "legacy_source": {
                "model_index": model_index,
                "scheduler_config_path": str(scheduler_cfg_path),
                "unet_config_path": str(unet_cfg_path),
            },
        },
        "__config_path__": str(model_index_path),
    }
    return cfg


def load_run_config(ckpt_dir: Path) -> dict:
    """
    load_run_config Function

    Loads the saved training config from a checkpoint directory.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.

    Outputs:
        - cfg: (dict) Parsed config with __config_path__ injected.
    """
    cfg_path = ckpt_dir / "train_config.json"
    if not cfg_path.exists():
        return _load_diffusers_legacy_run_config(ckpt_dir)
    cfg = load_json_config(cfg_path)
    existing_path = cfg.get("__config_path__")
    if existing_path:
        existing = Path(existing_path)
        if existing.exists():
            return cfg
    cfg["__config_path__"] = str(cfg_path)
    return cfg


def resolve_checkpoint(ckpt_dir: Path, model_type: str) -> Path:
    """
    resolve_checkpoint Function

    Resolves the best checkpoint path for a given model type.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.
        - model_type: (String) Model type name.

    Outputs:
        - path: (Path) Selected checkpoint path.
    """
    model_type = str(model_type).lower()
    candidates = []
    if model_type == "vae":
        candidates = ["vae_best.pt", "vae_last.pt"]
    elif model_type == "diffusion":
        candidates = ["diff_best.pt", "diff_last.pt"]
    elif model_type == "flow_matching":
        candidates = ["flow_best.pt", "flow_last.pt"]
    else:
        candidates = ["*.pt"]

    for name in candidates:
        path = ckpt_dir / name
        if path.exists():
            return path
    if model_type == "diffusion":
        legacy_unet_st = ckpt_dir / "unet" / "diffusion_pytorch_model.safetensors"
        if legacy_unet_st.exists():
            return legacy_unet_st
    if candidates == ["*.pt"]:
        pts = sorted(ckpt_dir.glob("*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _eval_cache_subdir(cache_subdir: str | None) -> str:
    cache_name = str(cache_subdir or "cache")
    return cache_name if cache_name.endswith("_eval") else f"{cache_name}_eval"


def build_sampling_dataset(cfg: dict, data_txt: str | None, evaluate: bool = False) -> object:
    """
    build_sampling_dataset Function

    Builds a dataset for sampling (test split by default, optional split override).

    Inputs:
        - cfg: (dict) Full config dict.
        - data_txt: (String | None) Optional split file override.
        - evaluate: (Boolean) If True, use the dataset test split and an eval cache namespace.

    Outputs:
        - dataset: (object) Dataset instance.
    """
    training_cfg = dict(cfg.get("training", {}))
    if evaluate:
        if data_txt:
            training_cfg["split_file"] = data_txt
        else:
            training_cfg.pop("split_file", None)
        training_cfg["tensor_cache_subdir"] = _eval_cache_subdir(training_cfg.get("tensor_cache_subdir"))
    elif data_txt:
        training_cfg["split_file"] = data_txt
    cfg_path = Path(cfg.get("__config_path__", "")) if cfg.get("__config_path__") else None
    return build_dataset_from_config(training_cfg, cfg.get("model", {}), train=False, cfg_path=cfg_path)


def resolve_output_root(ckpt_dir: Path, output_dir: str | None, save: bool) -> Path | None:
    """
    resolve_output_root Function

    Resolves output directory root for saved tensors.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.
        - output_dir: (String | None) Optional output override.
        - save: (Boolean) Whether to save outputs.

    Outputs:
        - output_root: (Path | None) Output directory or None.
    """
    if not save:
        return None
    if output_dir:
        return Path(output_dir)
    return ckpt_dir / "outputs"


def resolve_sample_indices(dataset, num_samples: int | None, seed: int = 42) -> list[int]:
    """
    Resolve a deterministic random subset of dataset indices.
    """
    total = len(dataset)
    if total == 0:
        return []
    if num_samples is None or int(num_samples) <= 0 or int(num_samples) >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(list(range(total)), int(num_samples))


def progress_batches(dataset, batch_size: int, desc: str, indices: list[int] | None = None):
    """
    Yield dataset batches with a tqdm progress bar when available.
    Falls back to plain iteration if tqdm is unavailable.
    """
    selected = list(range(len(dataset))) if indices is None else list(indices)
    total = len(selected)
    bs = max(int(batch_size), 1)
    total_batches = (total + bs - 1) // bs
    iterator = iter_batches(dataset, batch_size, indices=selected)
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(iterator, total=total_batches, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        pass
    return iterator


def build_tensor_cache_from_config(
    cfg: dict,
    data_txt: str | None,
    batch_size: int,
    seed: int,
    num_samples: int | None,
    desc: str = "build_tensor_cache",
    evaluate: bool = True,
) -> int:
    """
    Iterate dataset samples to trigger tensor cache reads/writes without model inference.
    """
    dataset = build_sampling_dataset(cfg, data_txt, evaluate=evaluate)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    total = 0
    for _, samples in progress_batches(dataset, batch_size, desc, indices=selected_indices):
        for sample in samples:
            _ = sample["target"]
            _ = sample.get("image")
        total += len(samples)
    return total


def append_eval_metrics(ckpt_dir: Path, row: dict) -> Path:
    """
    Append one evaluation summary row to ckpt_dir/eval_metrics.csv.
    """
    ckpt_dir = Path(ckpt_dir)
    out_path = ckpt_dir / "eval_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): str(v) for k, v in row.items()}
    fieldnames = list(payload.keys())
    exists = out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(payload)
    return out_path


def append_per_image_eval_metrics(ckpt_dir: Path, rows: list[dict]) -> Path:
    """
    Write per-sample evaluation rows to ckpt_dir/eval_metrics_per_image.csv.
    """
    ckpt_dir = Path(ckpt_dir)
    out_path = ckpt_dir / "eval_metrics_per_image.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not out_path.exists():
            out_path.write_text("")
        return out_path
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path


def run_self_tests() -> None:
    """
    Lightweight tests for sampling utility helpers.
    """
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg_path = root / "train_config.json"
        cfg = {"training": {"data_root": str(root)}, "model": {"model_type": "vae"}}
        cfg_path.write_text(json.dumps(cfg))
        loaded = load_run_config(root)
        assert "__config_path__" in loaded
        out = resolve_output_root(root, None, True)
        assert out == root / "outputs"
