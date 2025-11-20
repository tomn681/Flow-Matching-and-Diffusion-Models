"""
Callable VAE trainer that consumes a JSON config and trains the AutoencoderKL.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...models.vae import AutoencoderKL
from ...models.vae.losses import (
    PatchDiscriminator,
    PerceptualLoss,
    discriminator_hinge_loss,
    generator_hinge_loss,
    vq_regularizer,
)
from ...utils import DefaultDataset

HISTORY_FILENAME = "metrics.jsonl"


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config containing `training` and `vae` sections."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        cfg = json.load(fh)
    if "training" not in cfg or "vae" not in cfg:
        raise ValueError("Config must contain 'training' and 'vae' sections.")
    return cfg


def maybe_resize_to_resolution(tensor: torch.Tensor, resolution: int) -> tuple[torch.Tensor, bool]:
    """Resize spatial dims to a fixed resolution; returns (resized, resized_bool)."""
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


def maybe_resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Resize spatial dims to match a reference tensor; returns (resized, resized_bool)."""
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
    """Construct a dataloader for the LDCT dataset with default collate."""
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
    """Persist model/optimizer state for a given epoch."""
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
    """Run one training epoch and return averaged metrics plus updated global step."""
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
                bce_target = (inputs + 1.0) * 0.5  # map [-1, 1] -> [0, 1]
                recon_loss = F.binary_cross_entropy_with_logits(recon, bce_target)
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
    """Evaluate the model on a validation set and return averaged metrics."""
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
            bce_target = (inputs + 1.0) * 0.5  # map [-1, 1] -> [0, 1]
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


def default_collate_without_nones(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while preserving all-None entries for optional keys."""
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


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    merged = json.loads(json.dumps(cfg))  # shallow copy via json to avoid mutation
    for section in ("training", "vae"):
        if section in overrides and overrides[section]:
            merged[section].update({k: v for k, v in overrides[section].items() if v is not None})
    return merged


def train(config_path: Path | str, data_root: Path | str, *, overrides: dict[str, Any] | None = None) -> None:
    """End-to-end VAE training loop driven by JSON config, with optional overrides."""
    config_path = Path(config_path)
    data_root = Path(data_root)

    cfg = load_config(config_path)
    cfg = _apply_overrides(cfg, overrides)
    training_cfg: dict[str, Any] = cfg["training"]
    vae_cfg: dict[str, Any] = cfg["vae"]

    output_dir = Path(training_cfg["output_dir"])
    img_size = int(training_cfg.get("img_size") or vae_cfg.get("resolution"))
    device = torch.device(training_cfg.get("device", "cpu"))

    vae_cfg["attn_resolutions"] = tuple(vae_cfg.get("attn_resolutions", ()))
    vae_cfg["ch_mult"] = tuple(vae_cfg.get("ch_mult", ()))
    vae_cfg["resolution"] = img_size

    torch.manual_seed(int(training_cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training_cfg["seed"]))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    model = AutoencoderKL(**vae_cfg).to(device)
    perceptual_needed = training_cfg["recon_type"] == "perceptual" or training_cfg["perceptual_weight"] > 0
    perceptual = PerceptualLoss(resize=True).to(device) if perceptual_needed else None
    discriminator = (
        PatchDiscriminator(in_channels=vae_cfg["out_channels"]).to(device) if training_cfg["gan_weight"] > 0 else None
    )
    optimizer = AdamW(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    disc_optimizer = (
        AdamW(discriminator.parameters(), lr=training_cfg["disc_lr"], weight_decay=0.0) if discriminator is not None else None
    )

    resume_path = Path(training_cfg["resume"]) if training_cfg.get("resume") else None
    start_epoch = 1
    if resume_path is not None and resume_path.exists():
        logging.info("Resuming from checkpoint %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    scaler = torch.cuda.amp.GradScaler() if training_cfg["use_amp"] and device.type == "cuda" else None

    train_loader = make_dataloader(
        data_root,
        train=True,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        img_size=img_size,
        slice_count=training_cfg["slice_count"],
    )
    val_loader = make_dataloader(
        data_root,
        train=False,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        img_size=img_size,
        slice_count=training_cfg["slice_count"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "training": {**training_cfg, "data_root": str(data_root)},
        "vae": vae_cfg,
    }
    with (output_dir / "train_config.json").open("w") as fh:
        json.dump(resolved_config, fh, indent=2, default=str)

    history_path = output_dir / HISTORY_FILENAME
    if start_epoch <= 1 and not (resume_path and resume_path.exists()) and history_path.exists():
        history_path.unlink()

    best_val = float("inf")
    global_step = 0
    for epoch in range(start_epoch, int(training_cfg["epochs"]) + 1):
        train_metrics, global_step = train_epoch(
            model,
            discriminator,
            disc_optimizer,
            perceptual,
            train_loader,
            optimizer,
            device,
            scaler,
            reg_type=training_cfg["reg_type"],
            kl_weight=training_cfg["kl_weight"],
            kl_anneal_steps=training_cfg["kl_anneal_steps"],
            perceptual_weight=training_cfg["perceptual_weight"],
            recon_type=training_cfg["recon_type"],
            gan_weight=training_cfg["gan_weight"],
            gan_start=training_cfg["gan_start"],
            global_step=global_step,
            use_amp=training_cfg["use_amp"],
        )
        val_metrics = evaluate(
            model,
            perceptual,
            val_loader,
            device,
            reg_type=training_cfg["reg_type"],
            kl_weight=training_cfg["kl_weight"],
            perceptual_weight=training_cfg["perceptual_weight"],
            recon_type=training_cfg["recon_type"],
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

        if epoch % training_cfg["save_every"] == 0:
            ckpt_path = save_checkpoint(model, optimizer, epoch, output_dir, tag="ckpt")
            logging.info("Saved checkpoint to %s", ckpt_path)

        if is_best:
            best_val = val_metrics["loss"]
            ckpt_path = save_checkpoint(model, optimizer, epoch, output_dir, tag="best")
            logging.info("Updated best checkpoint: %s", ckpt_path)


__all__ = ["train"]
