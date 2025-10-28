"""
Simple training pipeline for the AutoencoderKL using the CT dataset utilities.

The configuration mirrors the Stable Diffusion VAE defaults (fm-boosting) but
is pared down for single-slice (`s_cnt=1`) training. Images are pulled from
`DefaultDataset` and only the standard-dose target volume is fed into the VAE.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..models.vae import AutoencoderKL
from ..utils import DefaultDataset


DEFAULT_CONFIG = dict(
    in_channels=1,
    out_channels=1,
    resolution=256,
    base_ch=128,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_resolutions=(),  # (16,),
    z_channels=4,
    embed_dim=4,
    dropout=0.0,
    use_attention=True,
    spatial_dims=2,
    emb_channels=None,
    use_scale_shift_norm=False,
    double_z=True,
)


def parse_list_argument(raw: str) -> Iterable[int]:
    if not raw:
        return ()
    if isinstance(raw, (tuple, list)):
        return tuple(int(v) for v in raw)
    return tuple(int(v.strip()) for v in raw.split(",") if v.strip())


def make_dataloader(
    data_root: Path,
    train: bool,
    batch_size: int,
    num_workers: int,
    img_size: int,
    slice_count: int,
    diff: bool = True,
    *,
    collate_fn=None,
) -> DataLoader:
    dataset = DefaultDataset(
        str(data_root),
        s_cnt=slice_count,
        img_size=img_size,
        train=train,
        diff=diff,
        norm=True,
    )
    if collate_fn is None:
        collate_fn = default_collate_without_nones
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collate_fn,
    )


def save_checkpoint(
    model: AutoencoderKL,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    out_dir: Path,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"vae_{tag}_epoch{epoch:04d}.pt"
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def train_epoch(
    model: AutoencoderKL,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    kl_weight: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    num_samples = 0

    autocast = torch.cuda.amp.autocast if use_amp else nullcontext

    for batch in dataloader:
        inputs = batch["target"].to(device, non_blocking=True)
        inputs = inputs * 2.0 - 1.0  # normalize to [-1, 1]
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            recon, posterior = model(inputs, sample_posterior=True)
            recon_loss = F.mse_loss(recon, inputs)
            kl_loss = posterior.kl().mean()
            loss = recon_loss + kl_weight * kl_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        totals["loss"] += loss.detach().item() * batch_size
        totals["recon"] += recon_loss.detach().item() * batch_size
        totals["kl"] += kl_loss.detach().item() * batch_size
        num_samples += batch_size

    return {k: v / max(1, num_samples) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: AutoencoderKL,
    dataloader: DataLoader,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    num_samples = 0

    for batch in dataloader:
        inputs = batch["target"].to(device, non_blocking=True)
        inputs = inputs * 2.0 - 1.0
        recon, posterior = model(inputs, sample_posterior=False)
        recon_loss = F.mse_loss(recon, inputs)
        kl_loss = posterior.kl().mean()
        loss = recon_loss + kl_weight * kl_loss

        batch_size = inputs.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["recon"] += recon_loss.item() * batch_size
        totals["kl"] += kl_loss.item() * batch_size
        num_samples += batch_size

    return {k: v / max(1, num_samples) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AutoencoderKL on LDCT data.")
    parser.add_argument("-d", "--data-root", type=Path, required=True, help="Dataset directory containing train/test splits.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("checkpoints/vae"), help="Where to store checkpoints and logs.")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-D", "--weight-decay", type=float, default=0.0)
    parser.add_argument("-k", "--kl-weight", type=float, default=1e-6, help="Scale factor for the KL divergence term.")
    parser.add_argument("-s", "--slice-count", type=int, default=1, help="Number of slices per sample (s_cnt).")
    parser.add_argument("-i", "--img-size", type=int, default=256, help="Side length to which slices are resized.")
    parser.add_argument("-a", "--attn-resolutions", type=str, default="16", help="Comma separated downsample factors where attention is enabled.")
    parser.add_argument("-c", "--channel-mult", type=str, default="1,2,4,4", help="Comma separated channel multipliers.")
    parser.add_argument("-I", "--in-channels", type=int, default=None, help="Number of input channels for the VAE (defaults to config).")
    parser.add_argument("-O", "--out-channels", type=int, default=None, help="Number of output channels for the VAE (defaults to config).")
    parser.add_argument("-v", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-A", "--use-amp", action="store_true", help="Enable automatic mixed precision (fp16).")
    parser.add_argument("-S", "--save-every", type=int, default=10, help="Epoch frequency for checkpointing.")
    parser.add_argument("-r", "--resume", type=Path, default=None, help="Path to an existing checkpoint to resume from.")
    parser.add_argument("-R", "--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    device = torch.device(args.device)
    attn_resolutions = parse_list_argument(args.attn_resolutions)
    channel_mult = parse_list_argument(args.channel_mult)

    model_kwargs = DEFAULT_CONFIG.copy()
    model_kwargs.update(
        dict(
            in_channels=args.in_channels if args.in_channels is not None else DEFAULT_CONFIG["in_channels"],
            out_channels=args.out_channels if args.out_channels is not None else DEFAULT_CONFIG["out_channels"],
            resolution=args.img_size,
            ch_mult=channel_mult or DEFAULT_CONFIG["ch_mult"],
            attn_resolutions=attn_resolutions or DEFAULT_CONFIG["attn_resolutions"],
        )
    )

    model = AutoencoderKL(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume is not None and args.resume.exists():
        logging.info("Resuming from checkpoint %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    train_loader = make_dataloader(
        args.data_root,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        slice_count=args.slice_count,
    )
    val_loader = make_dataloader(
        args.data_root,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        slice_count=args.slice_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_config.json").open("w") as fh:
        json.dump(
            {
                "args": vars(args),
                "model": model_kwargs,
            },
            fh,
            indent=2,
            default=str,
        )

    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            kl_weight=args.kl_weight,
            use_amp=args.use_amp,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            kl_weight=args.kl_weight,
        )

        logging.info(
            (
                "Epoch %03d | train_loss %.6f (recon %.6f, kl %.6f) "
                "| val_loss %.6f (recon %.6f, kl %.6f)"
            ),
            epoch,
            train_metrics["loss"],
            train_metrics["recon"],
            train_metrics["kl"],
            val_metrics["loss"],
            val_metrics["recon"],
            val_metrics["kl"],
        )

        if epoch % args.save_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, epoch, args.output_dir, tag="ckpt")
            logging.info("Saved checkpoint to %s", ckpt_path)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt_path = save_checkpoint(model, optimizer, epoch, args.output_dir, tag="best")
            logging.info("Updated best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
def default_collate_without_nones(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while keeping None entries untouched for keys like 'image'."""
    if not batch:
        return {}
    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [example[key] for example in batch]
        if all(value is None for value in values):
            collated[key] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Mixed None entries detected for key '{key}'.")
        first_non_none = next((value for value in values if value is not None), None)
        if isinstance(first_non_none, torch.Tensor):
            collated[key] = torch.stack(values)  # type: ignore[arg-type]
        else:
            collated[key] = values
    return collated
