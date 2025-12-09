"""
Lightweight training pipeline for modular VAEs built from JSON configs.
Accepts (dataset, json_path) and trains KL or VQ variants.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from time import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
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


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value, default: torch.device) -> torch.device:
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "none":
        return default
    return value if isinstance(value, torch.device) else torch.device(value)


def _log_model_summary(model: torch.nn.Module, vae_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> None:
    """
    Log model repr and parameter counts; optionally include torchinfo summary if available.
    """
    show = training_cfg.get("show_model_summary", True)
    if not show:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "Model architecture:",
        str(model),
        f"Parameters: total={_format_params(total_params)} ({total_params}), trainable={_format_params(trainable_params)} ({trainable_params})",
    ]

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
        lines.append("Model summary:")
        lines.append(str(info))
    except Exception as exc:  # pragma: no cover - optional path
        logging.debug("torchinfo summary skipped (%s)", exc)
    finally:
        for line in lines:
            logging.info(line)
            print(line, flush=True)


def _latent_shape(vae_cfg: Dict[str, Any]) -> tuple[int, ...]:
    spatial_dims = vae_cfg.get("spatial_dims", 2)
    embed_dim = vae_cfg["embed_dim"]
    resolution = vae_cfg["resolution"]
    ch_mult = tuple(vae_cfg["ch_mult"])
    factor = 2 ** (len(ch_mult) - 1)
    base_size = resolution // factor
    if spatial_dims == 3:
        return (embed_dim, base_size, base_size, base_size)
    if spatial_dims == 1:
        return (embed_dim, base_size)
    return (embed_dim, base_size, base_size)


def _make_grid(tensor: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    n, c, h, w = tensor.shape
    if n < rows * cols:
        raise ValueError(f"Need at least {rows*cols} images to build the grid, found {n}")
    tensor = tensor[: rows * cols]
    if c == 1:
        tensor = tensor.expand(-1, 3, h, w)
        c = 3
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.reshape(rows, cols, c, h, w)
    tensor = tensor.permute(2, 0, 3, 1, 4).contiguous()
    grid = tensor.reshape(c, rows * h, cols * w)
    grid_np = (grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    grid_np = np.transpose(grid_np, (1, 2, 0))
    return grid_np


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    logging.info("Saved grid: %s", path)


def _prepare_eval_batch(ds, count: int, device: torch.device) -> torch.Tensor:
    if ds is None or len(ds) == 0:
        raise RuntimeError("Dataset is empty; cannot prepare evaluation batch.")
    tensors = [ds[i]["target"] for i in range(min(len(ds), count))]
    if not tensors:
        raise RuntimeError("Failed to collect evaluation samples.")
    batch = torch.stack(tensors, dim=0).to(device)
    batch = batch * 2.0 - 1.0
    return batch


def train(dataset, json_path: Path | str, val_dataset=None) -> None:
    """
    Train a VAE on the given dataset using hyperparameters from JSON.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    cfg = _load_config(json_path)
    training_cfg = cfg["training"]
    _set_seed(training_cfg.get("seed"))

    device = torch.device(training_cfg.get("device", "cpu"))
    batch_size = int(training_cfg.get("batch_size", 4))
    allow_microbatching = bool(training_cfg.get("allow_microbatching", True))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    recon_type = training_cfg.get("recon_type", "l1")
    perceptual_weight = float(training_cfg.get("perceptual_weight", 0.0))
    gan_weight = float(training_cfg.get("gan_weight", 0.0))
    gan_start = int(training_cfg.get("gan_start", 0))
    kl_weight = float(training_cfg.get("kl_weight", 0.0))
    kl_anneal_steps = int(training_cfg.get("kl_anneal_steps", 0))
    codebook_weight = float(training_cfg.get("codebook_weight", 1.0))
    save_every = int(training_cfg.get("save_every", 1))
    output_dir = Path(training_cfg.get("output_dir", "checkpoints/vae"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_path = output_dir / "train_config.json"
    if not train_cfg_path.exists():
        with train_cfg_path.open("w") as fh:
            json.dump(cfg, fh, indent=2)
    best_metric = float("inf")

    model = VAEFactory().build_from_json(json_path).to(device)
    _log_model_summary(model, cfg["vae"], training_cfg)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _make_scheduler(optimizer, training_cfg)

    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    perceptual = PerceptualLoss(resize=True).to(device) if perceptual_weight > 0 else None
    perceptual_device = _resolve_device(training_cfg.get("perceptual_device"), device)
    if perceptual is not None:
        perceptual = perceptual.to(perceptual_device)

    discriminator = model.make_discriminator().to(device) if gan_weight > 0 else None
    disc_device = _resolve_device(training_cfg.get("disc_device"), device) if discriminator else device
    if discriminator:
        discriminator = discriminator.to(disc_device)
    disc_optimizer = AdamW(discriminator.parameters(), lr=training_cfg.get("disc_lr", lr)) if discriminator else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        if val_dataset is not None
        else None
    )

    data_line = "Data: train_samples=%d%s | batch_size=%d | micro_batching=%s | num_workers=%d" % (
        len(dataset),
        f", val_samples={len(val_dataset)}" if val_dataset is not None else "",
        batch_size,
        "enabled" if allow_microbatching else "disabled",
        num_workers,
    )
    logging.info(data_line)
    print(data_line, flush=True)

    sample_count = 20
    sample_dataset = val_dataset if val_dataset is not None else dataset
    sample_batch = _prepare_eval_batch(sample_dataset, sample_count, device)
    latent_shape = _latent_shape(cfg["vae"])
    sample_dir = output_dir / "samples"

    resume_flag = training_cfg.get("resume")
    if isinstance(resume_flag, str) and resume_flag.lower() == "none":
        resume_flag = None
    start_epoch = 1
    if resume_flag:
        if isinstance(resume_flag, str):
            ckpt_path = Path(resume_flag)
        else:
            candidates = list(output_dir.glob("vae_last.pt")) + list(output_dir.glob("vae_best.pt"))
            if not candidates:
                candidates = list(output_dir.glob("*.pt"))
            if not candidates:
                ckpt_path = None
            else:
                ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
        if ckpt_path and ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(payload.get("model", payload))
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "disc_optimizer" in payload and disc_optimizer and payload.get("disc_optimizer"):
                disc_optimizer.load_state_dict(payload["disc_optimizer"])
            if "scheduler" in payload and scheduler and payload.get("scheduler"):
                scheduler.load_state_dict(payload["scheduler"])
            if "scaler" in payload and scaler and payload.get("scaler"):
                scaler.load_state_dict(payload["scaler"])
            best_metric = payload.get("best_metric", best_metric)
            start_epoch = payload.get("epoch", 0) + 1
            logging.info("Resumed from %s (epoch %d)", ckpt_path, start_epoch - 1)
            print(f"Resumed from {ckpt_path} (epoch {start_epoch - 1})", flush=True)

    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if discriminator:
            discriminator.train()
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "perceptual": 0.0, "g_gan": 0.0, "d_gan": 0.0, "vq": 0.0}
        num_samples = 0
        train_loop = tqdm(dataloader, desc=f"Train {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        current_micro = batch_size
        warned_micro = False
        for batch in train_loop:
            batch_success = False
            while not batch_success:
                batch_start = time()
                inputs = batch["target"].to(device)
                inputs = inputs * 2.0 - 1.0
                bs = inputs.size(0)

                optimizer.zero_grad(set_to_none=True)
                if disc_optimizer:
                    disc_optimizer.zero_grad(set_to_none=True)

                try:
                    chunks = inputs.split(current_micro)
                    accum_steps = len(chunks)
                    d_loss_val = torch.tensor(0.0, device=device)

                    for chunk in chunks:
                        with autocast(device_type=device.type, enabled=use_amp):
                            if hasattr(model, "codebook"):
                                rec, vq_info = model(chunk)
                                vq_loss = vq_info["vq_loss"]
                                kl_term = torch.tensor(0.0, device=device)
                            else:
                                rec, posterior = model(chunk, sample_posterior=True)
                                vq_loss = torch.tensor(0.0, device=device)
                                kl_term = posterior.kl().mean()

                            if recon_type == "l1":
                                recon_loss = F.l1_loss(rec, chunk)
                            elif recon_type == "mse":
                                recon_loss = F.mse_loss(rec, chunk)
                            elif recon_type == "bce":
                                bce_target = (chunk + 1.0) * 0.5
                                recon_loss = F.binary_cross_entropy_with_logits(rec, bce_target)
                            else:
                                raise ValueError(f"Unsupported recon_type '{recon_type}'.")

                            if perceptual is not None:
                                rec_p = rec if rec.device == perceptual_device else rec.to(perceptual_device)
                                chunk_p = chunk if chunk.device == perceptual_device else chunk.to(perceptual_device)
                                perc_loss = perceptual(rec_p, chunk_p).to(device)
                            else:
                                perc_loss = torch.tensor(0.0, device=device)

                            disc_active = discriminator is not None and gan_weight > 0 and epoch >= gan_start
                            if disc_active:
                                rec_d = rec if rec.device == disc_device else rec.to(disc_device)
                                chunk_d = chunk if chunk.device == disc_device else chunk.to(disc_device)
                                fake_pred = discriminator(rec_d)
                                g_gan_loss = generator_hinge_loss(fake_pred).to(device)
                            else:
                                g_gan_loss = torch.tensor(0.0, device=device)

                            kl_scale = kl_weight
                            if kl_anneal_steps > 0:
                                step_for_anneal = max(1, global_step + 1)
                                kl_scale = kl_weight * min(1.0, step_for_anneal / max(1, kl_anneal_steps))

                            total_loss = recon_loss + perceptual_weight * perc_loss + kl_scale * kl_term + codebook_weight * vq_loss + gan_weight * g_gan_loss

                        if scaler.is_enabled():
                            scaler.scale(total_loss / accum_steps).backward()
                        else:
                            (total_loss / accum_steps).backward()

                        if disc_active:
                            with autocast(device_type=disc_device.type, enabled=use_amp):
                                rec_d = rec.detach() if rec.device == disc_device else rec.detach().to(disc_device)
                                chunk_d = chunk.detach() if chunk.device == disc_device else chunk.detach().to(disc_device)
                                real_pred = discriminator(chunk_d)
                                fake_pred_detached = discriminator(rec_d)
                                d_loss = discriminator_hinge_loss(real_pred, fake_pred_detached)
                            if scaler.is_enabled():
                                scaler.scale(d_loss / accum_steps).backward()
                            else:
                                (d_loss / accum_steps).backward()
                            d_loss_val = d_loss.detach()

                        chunk_bs = chunk.size(0)
                        totals["loss"] += total_loss.detach().item() * chunk_bs
                        totals["recon"] += recon_loss.detach().item() * chunk_bs
                        totals["perceptual"] += perc_loss.detach().item() * chunk_bs
                        totals["kl"] += kl_term.detach().item() * chunk_bs
                        totals["vq"] += vq_loss.detach().item() * chunk_bs
                        totals["g_gan"] += g_gan_loss.detach().item() * chunk_bs
                        totals["d_gan"] += d_loss_val.item() * chunk_bs

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        if disc_optimizer:
                            scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        if disc_optimizer:
                            disc_optimizer.step()

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
                    batch_success = True

                    if current_micro < batch_size and not warned_micro:
                        micro_batches = math.ceil(batch_size / current_micro)
                        warn_msg = (
                            f"\033[38;5;208mThe maximum batch size for the current configuration is {current_micro}; "
                            f"training with {micro_batches} micro batches of size {current_micro} for gradient accumulation.\033[0m"
                        )
                        logging.warning(warn_msg)
                        print(warn_msg, flush=True)
                        warned_micro = True
                    global_step += 1
                except RuntimeError as err:
                    if "out of memory" not in str(err).lower():
                        raise
                    if not allow_microbatching:
                        raise RuntimeError(
                            "The current batch size is too large, please set it to a lower value or enable microbatching."
                        ) from err
                    torch.cuda.empty_cache()
                    if current_micro <= 1:
                        raise
                    current_micro = max(1, current_micro // 2)
                    continue

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
                    chunks = inputs.split(min(current_micro, batch_size))
                    for chunk in chunks:
                        with autocast(device_type=device.type, enabled=use_amp):
                            if hasattr(model, "codebook"):
                                rec, vq_info = model(chunk)
                                vq_loss = vq_info["vq_loss"]
                                kl_term = torch.tensor(0.0, device=device)
                            else:
                                rec, posterior = model(chunk, sample_posterior=False)
                                vq_loss = torch.tensor(0.0, device=device)
                                kl_term = posterior.kl().mean()

                            if recon_type == "l1":
                                recon_loss = F.l1_loss(rec, chunk)
                            elif recon_type == "mse":
                                recon_loss = F.mse_loss(rec, chunk)
                            elif recon_type == "bce":
                                bce_target = (chunk + 1.0) * 0.5
                                recon_loss = F.binary_cross_entropy_with_logits(rec, bce_target)
                            else:
                                raise ValueError(f"Unsupported recon_type '{recon_type}'.")

                            if perceptual is not None:
                                rec_p = rec if rec.device == perceptual_device else rec.to(perceptual_device)
                                chunk_p = chunk if chunk.device == perceptual_device else chunk.to(perceptual_device)
                                perc_loss = perceptual(rec_p, chunk_p).to(device)
                            else:
                                perc_loss = torch.tensor(0.0, device=device)

                            disc_active = discriminator is not None and gan_weight > 0 and epoch >= gan_start
                            if disc_active:
                                rec_d = rec if rec.device == disc_device else rec.to(disc_device)
                                chunk_d = chunk if chunk.device == disc_device else chunk.to(disc_device)
                                fake_pred = discriminator(rec_d)
                                g_gan_loss = generator_hinge_loss(fake_pred).to(device)
                                d_loss_val = discriminator_hinge_loss(discriminator(chunk_d), discriminator(rec_d.detach())).to(device)
                            else:
                                g_gan_loss = torch.tensor(0.0, device=device)
                                d_loss_val = torch.tensor(0.0, device=device)

                            kl_scale = kl_weight
                            if kl_anneal_steps > 0:
                                step_for_anneal = max(1, global_step + 1)
                                kl_scale = kl_weight * min(1.0, step_for_anneal / max(1, kl_anneal_steps))

                            bs = chunk.size(0)
                            val_totals["loss"] += (
                                recon_loss + perceptual_weight * perc_loss + kl_scale * kl_term + codebook_weight * vq_loss + gan_weight * g_gan_loss
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

        current_metric = val_avg["loss"] if val_loader is not None else averaged["loss"]
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_metric": best_metric,
        }
        should_save = epoch % save_every == 0 or epoch == epochs
        if should_save:
            ckpt_path = output_dir / "vae_last.pt"
            torch.save(state, ckpt_path)
            logging.info("Saved checkpoint: %s", ckpt_path)
        if current_metric < best_metric:
            best_metric = current_metric
            state["best_metric"] = best_metric
            best_path = output_dir / "vae_best.pt"
            torch.save(state, best_path)
            logging.info("New best (%.6f) -> %s", best_metric, best_path)

        if should_save:
            model.eval()
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(sample_batch, sample_posterior=False) if not hasattr(model, "codebook") else model(sample_batch)
                rec = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                rec_grid = _make_grid(rec, 4, 5)
                noise = torch.randn((sample_count, *latent_shape), device=device)
                with autocast(device_type=device.type, enabled=use_amp):
                    gen = model.decode(noise)
                gen_grid = _make_grid(gen, 4, 5)
            _save_image(rec_grid, (sample_dir / "recon") / f"epoch{epoch:04d}.png")
            _save_image(gen_grid, (sample_dir / "gen") / f"epoch{epoch:04d}.png")
            model.train()
