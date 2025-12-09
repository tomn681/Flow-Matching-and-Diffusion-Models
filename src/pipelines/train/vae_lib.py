"""
Lightweight training pipeline for modular VAEs built from JSON configs.
Accepts (dataset, json_path) and trains KL or VQ variants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader

from models import VAEFactory
from nn.losses.vae import PerceptualLoss, PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss


def _load_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    sched_cfg = cfg.get("scheduler")
    if not sched_cfg:
        return None
    name = (sched_cfg.get("name") or "").lower()
    params = sched_cfg.get("params", {})
    if name == "steplr":
        return StepLR(optimizer, **params)
    if name == "cosineannealinglr":
        return CosineAnnealingLR(optimizer, **params)
    if name == "exponentiallr":
        return ExponentialLR(optimizer, **params)
    if name == "":
        return None
    raise ValueError(f"Unsupported scheduler '{name}'.")


def _format_params(count: int) -> str:
    if count >= 1e6:
        return f"{count/1e6:.2f}M"
    if count >= 1e3:
        return f"{count/1e3:.2f}K"
    return str(count)


def _log_model_summary(model: torch.nn.Module, vae_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> None:
    """
    Log model repr and parameter counts; optionally include torchinfo summary if available.
    """
    show = training_cfg.get("show_model_summary", True)
    if not show:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model architecture:\n%s", model)
    logging.info("Parameters: total=%s (%d), trainable=%s (%d)", _format_params(total_params), total_params, _format_params(trainable_params), trainable_params)

    try:
        from torchinfo import summary  # type: ignore

        spatial_dims = vae_cfg.get("spatial_dims", 2)
        in_ch = vae_cfg.get("in_channels", 3)
        res = vae_cfg.get("resolution", 256)
        if spatial_dims == 3:
            input_size = (1, in_ch, res, res, res)
        elif spatial_dims == 1:
            input_size = (1, in_ch, res)
        else:
            input_size = (1, in_ch, res, res)

        info = summary(
            model,
            input_size=input_size,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            depth=3,
            verbose=0,
        )
        logging.info("Model summary:\n%s", info)
    except Exception as exc:  # pragma: no cover - optional path
        logging.debug("torchinfo summary skipped (%s)", exc)


def train(dataset, json_path: Path | str, val_dataset=None) -> None:
    """
    Train a VAE on the given dataset using hyperparameters from JSON.
    """
    cfg = _load_config(json_path)
    training_cfg = cfg["training"]

    device = torch.device(training_cfg.get("device", "cpu"))
    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    recon_type = training_cfg.get("recon_type", "l1")
    perceptual_weight = float(training_cfg.get("perceptual_weight", 0.0))
    gan_weight = float(training_cfg.get("gan_weight", 0.0))
    gan_start = int(training_cfg.get("gan_start", 0))
    kl_weight = float(training_cfg.get("kl_weight", 0.0))
    codebook_weight = float(training_cfg.get("codebook_weight", 1.0))

    model = VAEFactory().build_from_json(json_path).to(device)
    _log_model_summary(model, cfg["vae"], training_cfg)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _make_scheduler(optimizer, training_cfg)

    perceptual = PerceptualLoss(resize=True).to(device) if perceptual_weight > 0 else None
    discriminator = model.make_discriminator().to(device) if gan_weight > 0 else None
    disc_optimizer = AdamW(discriminator.parameters(), lr=training_cfg.get("disc_lr", lr)) if discriminator else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        if val_dataset is not None
        else None
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    for epoch in range(1, epochs + 1):
        model.train()
        if discriminator:
            discriminator.train()
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "perceptual": 0.0, "g_gan": 0.0, "d_gan": 0.0, "vq": 0.0}
        num_samples = 0
        train_loop = tqdm(dataloader, desc=f"Train {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in train_loop:
            batch_start = time()
            inputs = batch["target"].to(device)
            inputs = inputs * 2.0 - 1.0

            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer:
                disc_optimizer.zero_grad(set_to_none=True)

            if hasattr(model, "codebook"):
                rec, vq_info = model(inputs)
                vq_loss = vq_info["vq_loss"]
                kl_term = torch.tensor(0.0, device=device)
            else:
                rec, posterior = model(inputs, sample_posterior=True)
                vq_loss = torch.tensor(0.0, device=device)
                kl_term = posterior.kl().mean()

            if recon_type == "l1":
                recon_loss = F.l1_loss(rec, inputs)
            elif recon_type == "mse":
                recon_loss = F.mse_loss(rec, inputs)
            elif recon_type == "bce":
                bce_target = (inputs + 1.0) * 0.5
                recon_loss = F.binary_cross_entropy_with_logits(rec, bce_target)
            else:
                raise ValueError(f"Unsupported recon_type '{recon_type}'.")

            perc_loss = perceptual(rec, inputs).to(device) if perceptual is not None else torch.tensor(0.0, device=device)

            disc_active = discriminator is not None and gan_weight > 0 and epoch >= gan_start
            if disc_active:
                fake_pred = discriminator(rec)
                g_gan_loss = generator_hinge_loss(fake_pred).to(device)
            else:
                g_gan_loss = torch.tensor(0.0, device=device)

            total_loss = recon_loss + perceptual_weight * perc_loss + kl_weight * kl_term + codebook_weight * vq_loss + gan_weight * g_gan_loss
            total_loss.backward()

            d_loss_val = torch.tensor(0.0, device=device)
            if disc_active:
                real_pred = discriminator(inputs)
                fake_pred_detached = discriminator(rec.detach())
                d_loss = discriminator_hinge_loss(real_pred, fake_pred_detached)
                d_loss.backward()
                d_loss_val = d_loss.detach()

            optimizer.step()
            if disc_optimizer:
                disc_optimizer.step()

            bs = inputs.size(0)
            totals["loss"] += total_loss.detach().item() * bs
            totals["recon"] += recon_loss.detach().item() * bs
            totals["perceptual"] += perc_loss.detach().item() * bs
            totals["kl"] += kl_term.detach().item() * bs
            totals["vq"] += vq_loss.detach().item() * bs
            totals["g_gan"] += g_gan_loss.detach().item() * bs
            totals["d_gan"] += d_loss_val.item() * bs
            num_samples += bs
            batch_time = time() - batch_start
            postfix = {
                "loss": f"{totals['loss']/max(1,num_samples):.4f}",
                "recon": f"{totals['recon']/max(1,num_samples):.4f}",
            }
            if not hasattr(model, "codebook"):
                postfix["kl"] = f"{totals['kl']/max(1,num_samples):.4f}"
            else:
                postfix["vq"] = f"{totals['vq']/max(1,num_samples):.4f}"
            if perceptual_weight > 0:
                postfix["perc"] = f"{totals['perceptual']/max(1,num_samples):.4f}"
            if gan_weight > 0:
                postfix["g_gan"] = f"{totals['g_gan']/max(1,num_samples):.4f}"
                postfix["d_gan"] = f"{totals['d_gan']/max(1,num_samples):.4f}"
            postfix["bt"] = f"{batch_time:.3f}s"
            train_loop.set_postfix(**postfix)

        averaged = {k: v / max(1, num_samples) for k, v in totals.items()}
        logging.info(
            "Epoch %03d | loss %.6f (recon %.6f, perc %.6f, kl %.6f, vq %.6f, g_gan %.6f, d_gan %.6f)",
            epoch,
            averaged["loss"],
            averaged["recon"],
            averaged["perceptual"],
            averaged["kl"],
            averaged["vq"],
            averaged["g_gan"],
            averaged["d_gan"],
        )

        if val_loader is not None:
            model.eval()
            if discriminator:
                discriminator.eval()
            val_totals = {k: 0.0 for k in totals}
            val_samples = 0
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", leave=False, dynamic_ncols=True)
                for batch in val_loop:
                    batch_start = time()
                    inputs = batch["target"].to(device)
                    inputs = inputs * 2.0 - 1.0
                    if hasattr(model, "codebook"):
                        rec, vq_info = model(inputs)
                        vq_loss = vq_info["vq_loss"]
                        kl_term = torch.tensor(0.0, device=device)
                    else:
                        rec, posterior = model(inputs, sample_posterior=False)
                        vq_loss = torch.tensor(0.0, device=device)
                        kl_term = posterior.kl().mean()

                    if recon_type == "l1":
                        recon_loss = F.l1_loss(rec, inputs)
                    elif recon_type == "mse":
                        recon_loss = F.mse_loss(rec, inputs)
                    elif recon_type == "bce":
                        bce_target = (inputs + 1.0) * 0.5
                        recon_loss = F.binary_cross_entropy_with_logits(rec, bce_target)
                    else:
                        raise ValueError(f"Unsupported recon_type '{recon_type}'.")

                    perc_loss = perceptual(rec, inputs).to(device) if perceptual is not None else torch.tensor(0.0, device=device)

                    disc_active = discriminator is not None and gan_weight > 0 and epoch >= gan_start
                    if disc_active:
                        fake_pred = discriminator(rec)
                        g_gan_loss = generator_hinge_loss(fake_pred).to(device)
                        d_loss_val = discriminator_hinge_loss(discriminator(inputs), discriminator(rec.detach()))
                    else:
                        g_gan_loss = torch.tensor(0.0, device=device)
                        d_loss_val = torch.tensor(0.0, device=device)

                    bs = inputs.size(0)
                    val_totals["loss"] += (
                        recon_loss + perceptual_weight * perc_loss + kl_weight * kl_term + codebook_weight * vq_loss + gan_weight * g_gan_loss
                    ).detach().item() * bs
                    val_totals["recon"] += recon_loss.detach().item() * bs
                    val_totals["perceptual"] += perc_loss.detach().item() * bs
                    val_totals["kl"] += kl_term.detach().item() * bs
                    val_totals["vq"] += vq_loss.detach().item() * bs
                    val_totals["g_gan"] += g_gan_loss.detach().item() * bs
                    val_totals["d_gan"] += d_loss_val.detach().item() * bs
                    val_samples += bs
                    batch_time = time() - batch_start
                    postfix = {
                        "loss": f"{val_totals['loss']/max(1,val_samples):.4f}",
                        "recon": f"{val_totals['recon']/max(1,val_samples):.4f}",
                    }
                    if not hasattr(model, "codebook"):
                        postfix["kl"] = f"{val_totals['kl']/max(1,val_samples):.4f}"
                    else:
                        postfix["vq"] = f"{val_totals['vq']/max(1,val_samples):.4f}"
                    if perceptual_weight > 0:
                        postfix["perc"] = f"{val_totals['perceptual']/max(1,val_samples):.4f}"
                    if gan_weight > 0:
                        postfix["g_gan"] = f"{val_totals['g_gan']/max(1,val_samples):.4f}"
                        postfix["d_gan"] = f"{val_totals['d_gan']/max(1,val_samples):.4f}"
                    postfix["bt"] = f"{batch_time:.3f}s"
                    val_loop.set_postfix(**postfix)
            val_avg = {k: v / max(1, val_samples) for k, v in val_totals.items()}
            logging.info(
                "Epoch %03d | val_loss %.6f (recon %.6f, perc %.6f, kl %.6f, vq %.6f, g_gan %.6f, d_gan %.6f)",
                epoch,
                val_avg["loss"],
                val_avg["recon"],
                val_avg["perceptual"],
                val_avg["kl"],
                val_avg["vq"],
                val_avg["g_gan"],
                val_avg["d_gan"],
            )

        if scheduler is not None:
            scheduler.step()
