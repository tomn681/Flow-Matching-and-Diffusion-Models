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
from ..models.vae.losses import (
    PatchDiscriminator,
    PerceptualLoss,
    discriminator_hinge_loss,
    generator_hinge_loss,
    vq_regularizer,
)
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

HISTORY_FILENAME = "metrics.jsonl"


def maybe_resize_to_resolution(
    tensor: torch.Tensor,
    resolution: int,
) -> tuple[torch.Tensor, bool]:
    """
    Resize spatial dimensions to match the requested resolution.
    Returns the resized tensor and a flag indicating whether a resize occurred.
    """
    spatial_dims = tensor.ndim - 2
    if spatial_dims <= 0:
        return tensor, False

    desired_shape = (resolution,) * spatial_dims
    if tensor.shape[-spatial_dims:] == desired_shape:
        return tensor, False

    mode_map = {1: "linear", 2: "bilinear", 3: "trilinear"}
    mode = mode_map.get(spatial_dims, "bilinear")

    resized = F.interpolate(tensor, size=desired_shape, mode=mode, align_corners=False)
    return resized, True


def maybe_resize_like(
    tensor: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """
    Resize `tensor` so its spatial dimensions match `reference`.
    Returns the resized tensor and a flag indicating whether resizing occurred.
    """
    spatial_dims = tensor.ndim - 2
    if spatial_dims <= 0:
        return tensor, False

    target_size = reference.shape[-spatial_dims:]
    if tensor.shape[-spatial_dims:] == target_size:
        return tensor, False

    mode_map = {1: "linear", 2: "bilinear", 3: "trilinear"}
    mode = mode_map.get(spatial_dims, "bilinear")
    resized = F.interpolate(tensor, size=target_size, mode=mode, align_corners=False)
    return resized, True


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
    discriminator: PatchDiscriminator | None,
    disc_optimizer: torch.optim.Optimizer | None,
    perceptual: PerceptualLoss | None,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    reg_type: str,
    kl_weight: float,
    kl_anneal_steps: int,
    perceptual_weight: float,
    recon_type: str,
    gan_weight: float,
    gan_start: int,
    global_step: int,
    use_amp: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "perceptual": 0.0, "g_gan": 0.0, "d_gan": 0.0}
    num_samples = 0
    perceptual_missing_warned = False

    autocast = torch.cuda.amp.autocast if use_amp else nullcontext

    resolution = getattr(getattr(model, "decoder", None), "resolution", None)
    resize_logged = False
    recon_resize_logged = False

    for batch in dataloader:
        inputs = batch["target"].to(device, non_blocking=True)
        inputs = inputs * 2.0 - 1.0  # normalize to [-1, 1]

        if resolution is not None:
            original_shape = inputs.shape
            inputs, resized = maybe_resize_to_resolution(inputs, resolution)
            if resized and not resize_logged:
                logging.warning(
                    "Resized training targets from %s to %s to match VAE resolution=%d.",
                    tuple(original_shape[-(inputs.ndim - 2):]),
                    tuple(inputs.shape[-(inputs.ndim - 2):]),
                    resolution,
                )
                resize_logged = True

        optimizer.zero_grad(set_to_none=True)
        if disc_optimizer is not None:
            disc_optimizer.zero_grad(set_to_none=True)

        with autocast():
            recon, posterior = model(inputs, sample_posterior=True)
            original_recon_shape = recon.shape
            recon, recon_resized = maybe_resize_like(recon, inputs)
            if recon_resized and not recon_resize_logged:
                logging.warning(
                    "Resized training reconstructions from %s to %s to match target size.",
                    tuple(original_recon_shape[-(recon.ndim - 2):]),
                    tuple(inputs.shape[-(inputs.ndim - 2):]),
                )
                recon_resize_logged = True
            if recon_type == "mse":
                recon_loss = F.mse_loss(recon, inputs)
            elif recon_type == "bce":
                recon_loss = F.binary_cross_entropy_with_logits(recon, inputs)
            else:  # perceptual
                if perceptual is None:
                    if not perceptual_missing_warned:
                        logging.warning(
                            "Perceptual loss selected but torchvision not available; falling back to MSE."
                        )
                        perceptual_missing_warned = True
                    recon_loss = F.mse_loss(recon, inputs)
                else:
                    recon_loss = perceptual(recon, inputs)
            perc_loss = perceptual(recon, inputs) if perceptual is not None and perceptual_weight > 0 else torch.tensor(0.0, device=device)

            if reg_type == "vq":
                reg_loss = vq_regularizer(posterior.mode())
            else:
                reg_loss = posterior.kl().mean()

            if kl_anneal_steps > 0:
                reg_weight = kl_weight * min(1.0, global_step / float(kl_anneal_steps))
            else:
                reg_weight = kl_weight

            g_loss = recon_loss + perceptual_weight * perc_loss + reg_weight * reg_loss

            disc_active = (
                discriminator is not None
                and disc_optimizer is not None
                and gan_weight > 0.0
                and global_step >= gan_start
            )
            if disc_active:
                fake_pred = discriminator(recon)
                g_gan_loss = generator_hinge_loss(fake_pred)
                g_loss = g_loss + gan_weight * g_gan_loss
            else:
                g_gan_loss = torch.tensor(0.0, device=device)

        if scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            g_loss.backward()
            optimizer.step()

        d_loss_val = torch.tensor(0.0, device=device)
        if disc_active:
            with autocast():
                real_pred = discriminator(inputs)
                fake_pred_detached = discriminator(recon.detach())
                d_loss = discriminator_hinge_loss(real_pred, fake_pred_detached)
            d_loss_val = d_loss.detach()
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(disc_optimizer)  # type: ignore[arg-type]
                scaler.update()
            else:
                d_loss.backward()
                disc_optimizer.step()  # type: ignore[union-attr]

        batch_size = inputs.size(0)
        totals["loss"] += g_loss.detach().item() * batch_size
        totals["recon"] += recon_loss.detach().item() * batch_size
        totals["kl"] += reg_loss.detach().item() * batch_size
        totals["perceptual"] += perc_loss.detach().item() * batch_size
        totals["g_gan"] += g_gan_loss.detach().item() * batch_size
        totals["d_gan"] += d_loss_val.item() * batch_size
        num_samples += batch_size
        global_step += 1

    return {k: v / max(1, num_samples) for k, v in totals.items()}, global_step


@torch.no_grad()
def evaluate(
    model: AutoencoderKL,
    perceptual: PerceptualLoss | None,
    dataloader: DataLoader,
    device: torch.device,
    reg_type: str,
    kl_weight: float,
    perceptual_weight: float,
    recon_type: str,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "perceptual": 0.0}
    num_samples = 0
    perceptual_missing_warned = False

    resolution = getattr(getattr(model, "decoder", None), "resolution", None)
    resize_logged = False
    recon_resize_logged = False

    for batch in dataloader:
        inputs = batch["target"].to(device, non_blocking=True)
        inputs = inputs * 2.0 - 1.0

        if resolution is not None:
            original_shape = inputs.shape
            inputs, resized = maybe_resize_to_resolution(inputs, resolution)
            if resized and not resize_logged:
                logging.warning(
                    "Resized validation targets from %s to %s to match VAE resolution=%d.",
                    tuple(original_shape[-(inputs.ndim - 2):]),
                    tuple(inputs.shape[-(inputs.ndim - 2):]),
                    resolution,
                )
                resize_logged = True

        recon, posterior = model(inputs, sample_posterior=False)
        original_recon_shape = recon.shape
        recon, recon_resized = maybe_resize_like(recon, inputs)
        if recon_resized and not recon_resize_logged:
            logging.warning(
                "Resized validation reconstructions from %s to %s to match target size.",
                tuple(original_recon_shape[-(recon.ndim - 2):]),
                tuple(inputs.shape[-(inputs.ndim - 2):]),
            )
            recon_resize_logged = True
        if recon_type == "mse":
            recon_loss = F.mse_loss(recon, inputs)
        elif recon_type == "bce":
            bce_target = (inputs + 1.0) * 0.5  # map to [0, 1]
            recon_loss = F.binary_cross_entropy_with_logits(recon, bce_target)
        else:
            if perceptual is None:
                if not perceptual_missing_warned:
                    logging.warning(
                        "Perceptual loss selected but torchvision not available; falling back to MSE."
                    )
                    perceptual_missing_warned = True
                recon_loss = F.mse_loss(recon, inputs)
            else:
                recon_loss = perceptual(recon, inputs)
        perc_loss = perceptual(recon, inputs) if perceptual is not None and perceptual_weight > 0 else torch.tensor(0.0, device=device)
        if reg_type == "vq":
            kl_loss = vq_regularizer(posterior.mode())
        else:
            kl_loss = posterior.kl().mean()
        loss = recon_loss + perceptual_weight * perc_loss + kl_weight * kl_loss

        batch_size = inputs.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["recon"] += recon_loss.item() * batch_size
        totals["kl"] += kl_loss.item() * batch_size
        totals["perceptual"] += perc_loss.item() * batch_size
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
    parser.add_argument("-k", "--kl-weight", type=float, default=1e-6, help="Scale factor for the regularization term (KL or VQ).")
    parser.add_argument("--kl-anneal-steps", type=int, default=0, help="Linear warmup steps for the KL/VQ regularizer.")
    parser.add_argument("--reg-type", type=str, choices=("kl", "vq"), default="kl", help="Regularization type: KL or VQ-style.")
    parser.add_argument("--recon-type", type=str, choices=("perceptual", "mse", "bce"), default="perceptual", help="Reconstruction loss type.")
    parser.add_argument("--perceptual-weight", type=float, default=0.0, help="Weight for an optional auxiliary perceptual term.")
    parser.add_argument("--gan-weight", type=float, default=1e-3, help="Weight for the generator adversarial loss term.")
    parser.add_argument("--gan-start", type=int, default=0, help="Number of steps to wait before enabling GAN loss.")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Learning rate for the discriminator.")
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
    parser.add_argument("--base-ch", type=int, default=None, help="Base channel count for the first encoder layer.")
    parser.add_argument("--num-res-blocks", type=int, default=None, help="Residual blocks per resolution level.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability applied inside residual blocks.")
    parser.add_argument("--z-channels", type=int, default=None, help="Latent channel count before quantisation.")
    parser.add_argument("--embed-dim", type=int, default=None, help="Latent embedding dimension after the quant conv.")
    parser.add_argument("--attn-heads", type=int, default=None, help="Number of attention heads when attention is enabled.")
    parser.add_argument("--attn-dim-head", type=int, default=None, help="Dimensionality per attention head.")
    parser.add_argument("--spatial-dims", type=int, choices=(1, 2, 3), default=None, help="Spatial dimensionality (1, 2, or 3D).")
    parser.add_argument("--no-attention", action="store_true", help="Disable spatial attention blocks regardless of resolution settings.")
    parser.add_argument("--scale-shift-norm", action="store_true", help="Enable scale-shift (FiLM) conditioning in residual blocks.")
    parser.add_argument("--single-z", action="store_true", help="Disable double_z so the encoder outputs only z_channels.")
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

    override_fields = {
        "base_ch": args.base_ch,
        "num_res_blocks": args.num_res_blocks,
        "dropout": args.dropout,
        "z_channels": args.z_channels,
        "embed_dim": args.embed_dim,
        "attn_heads": args.attn_heads,
        "attn_dim_head": args.attn_dim_head,
        "spatial_dims": args.spatial_dims,
    }
    for key, value in override_fields.items():
        if value is not None:
            model_kwargs[key] = value

    if args.no_attention:
        model_kwargs["use_attention"] = False
    if args.scale_shift_norm:
        model_kwargs["use_scale_shift_norm"] = True
    if args.single_z:
        model_kwargs["double_z"] = False

    model = AutoencoderKL(**model_kwargs).to(device)
    perceptual_needed = args.recon_type == "perceptual" or args.perceptual_weight > 0
    perceptual = PerceptualLoss(resize=True).to(device) if perceptual_needed else None
    discriminator = (
        PatchDiscriminator(in_channels=model_kwargs["out_channels"]).to(device) if args.gan_weight > 0 else None
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    disc_optimizer = (
        AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0) if discriminator is not None else None
    )

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

    history_path = args.output_dir / HISTORY_FILENAME
    if start_epoch <= 1 and not (args.resume and args.resume.exists()) and history_path.exists():
        history_path.unlink()

    best_val = float("inf")
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, global_step = train_epoch(
            model,
            discriminator,
            disc_optimizer,
            perceptual,
            train_loader,
            optimizer,
            device,
            scaler,
            kl_weight=args.kl_weight,
            perceptual_weight=args.perceptual_weight,
            gan_weight=args.gan_weight,
            gan_start=args.gan_start,
            global_step=global_step,
            use_amp=args.use_amp,
        )
        val_metrics = evaluate(
            model,
            perceptual,
            val_loader,
            device,
            kl_weight=args.kl_weight,
            perceptual_weight=args.perceptual_weight,
        )

        logging.info(
            (
                "Epoch %03d | train_loss %.6f (recon %.6f, perc %.6f, kl %.6f, g_gan %.6f, d_gan %.6f) "
                "| val_loss %.6f (recon %.6f, perc %.6f, kl %.6f)"
            ),
            epoch,
            train_metrics["loss"],
            train_metrics["recon"],
            train_metrics["perceptual"],
            train_metrics["kl"],
            train_metrics["g_gan"],
            train_metrics["d_gan"],
            val_metrics["loss"],
            val_metrics["recon"],
            val_metrics["perceptual"],
            val_metrics["kl"],
        )

        is_best = val_metrics["loss"] < best_val

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best": is_best,
        }
        with history_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        if epoch % args.save_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, epoch, args.output_dir, tag="ckpt")
            logging.info("Saved checkpoint to %s", ckpt_path)

        if is_best:
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
