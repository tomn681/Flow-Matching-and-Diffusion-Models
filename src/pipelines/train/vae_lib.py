"""
Lightweight training pipeline for modular VAEs built from JSON configs.
Accepts (dataset, json_path) and trains KL or VQ variants.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader

from utils.model_utils.vae_utils import build_vae_model
from nn.losses.vae import PerceptualLoss, PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss, focal_loss, bce_focal_loss
from utils.dataset_utils import save_output_tensor
import utils


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


def _disc_is_active(
    discriminator: torch.nn.Module | None,
    gan_weight: float,
    gan_start: int,
    gan_start_steps: int | None,
    epoch: int,
    global_step: int,
) -> bool:
    if discriminator is None or gan_weight <= 0:
        return False
    if gan_start_steps is not None:
        return global_step >= gan_start_steps
    return epoch >= gan_start


def train(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    """
    Train a VAE on the given dataset using hyperparameters from JSON.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    cfg = utils.load_json_config(json_path)
    training_cfg = cfg["training"]
    utils.set_seed(training_cfg.get("seed"))

    manual_device = training_cfg.get("manual_device")
    if manual_device is None or (isinstance(manual_device, str) and manual_device.lower() == "none"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(manual_device)
    batch_size = int(training_cfg.get("batch_size", 4))
    allow_microbatching = bool(training_cfg.get("allow_microbatching", True))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    reg_type = str(training_cfg.get("reg_type", "kl")).lower()
    recon_type = training_cfg.get("recon_type", "l1")
    perceptual_weight = float(training_cfg.get("perceptual_weight", 0.0))
    gan_weight = float(training_cfg.get("gan_weight", 0.0))
    gan_start = int(training_cfg.get("gan_start", 0))
    gan_start_steps = training_cfg.get("gan_start_steps")
    if gan_start_steps is not None:
        gan_start_steps = int(gan_start_steps)
    kl_weight = float(training_cfg.get("kl_weight", 0.0))
    kl_anneal_steps = int(training_cfg.get("kl_anneal_steps", 0))
    codebook_weight = float(training_cfg.get("codebook_weight", 1.0))
    save_every = int(training_cfg.get("save_every", 1))
    base_output_dir = Path(training_cfg.get("output_dir", "checkpoints/vae"))
    output_dir = utils.allocate_run_dir(base_output_dir) if resume is None else base_output_dir
    training_cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_path = output_dir / "train_config.json"
    if not train_cfg_path.exists():
        utils.save_json_config(train_cfg_path, cfg)
    best_metric = float("inf")
    metrics_path = output_dir / "metrics.csv"
    metrics_keys = ["loss", "recon"]
    if reg_type == "kl" or kl_weight > 0:
        metrics_keys.append("kl")
    if reg_type == "vq" or codebook_weight > 0:
        metrics_keys.append("vq")
    if perceptual_weight > 0:
        metrics_keys.append("perceptual")
    if gan_weight > 0:
        metrics_keys.extend(["g_gan", "d_gan"])
    if utils.is_main_process() and not metrics_path.exists():
        header = "epoch," + ",".join(metrics_keys) + "\n"
        metrics_path.write_text(header)

    model = build_vae_model(cfg, device, ckpt_path=None, set_eval=False)
    model_cfg = cfg.get("model", {})
    latent_type = str(model_cfg.get("latent_type", "kl")).lower()
    codebook_active = latent_type == "vq" or reg_type == "vq"
    effective_codebook_weight = codebook_weight if codebook_active else 0.0
    utils.summarize_model(model, model_cfg, training_cfg)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _make_scheduler(optimizer, training_cfg)

    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    perceptual = PerceptualLoss(resize=True).to(device) if perceptual_weight > 0 else None
    perceptual_device = utils.resolve_device(training_cfg.get("perceptual_device"), device)
    if perceptual is not None:
        perceptual = perceptual.to(perceptual_device)

    discriminator = model.make_discriminator().to(device) if gan_weight > 0 else None
    disc_device = utils.resolve_device(training_cfg.get("disc_device"), device) if discriminator else device
    if discriminator:
        discriminator = discriminator.to(disc_device)
    disc_optimizer = AdamW(discriminator.parameters(), lr=training_cfg.get("disc_lr", lr)) if discriminator else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        if val_dataset is not None
        else None
    )

    data_line = "Data: train_samples=%d%s | batch_size=%d | micro_batching=%s | num_workers=%d | epochs=%d" % (
        len(dataset),
        f", val_samples={len(val_dataset)}" if val_dataset is not None else "",
        batch_size,
        "enabled" if allow_microbatching else "disabled",
        num_workers,
        epochs,
    )
    spaced_data_line = f"\n\033[36m{data_line}\033[0m\n"
    logging.info(data_line)
    print(spaced_data_line, flush=True)

    sample_count = int(training_cfg.get("visual_samples", 20))
    visual_enabled = bool(training_cfg.get("save_images", True))
    visual_every = int(training_cfg.get("save_images_every", 1))
    sample_dataset = val_dataset if val_dataset is not None else dataset
    sample_batch = utils.prepare_eval_batch(sample_dataset, sample_count, device, seed=training_cfg.get("seed"))
    latent_shape_ = utils.latent_shape(model_cfg)
    sample_dir = output_dir / "samples"

    resume_flag = resume if resume is not None else training_cfg.get("resume")
    if isinstance(resume_flag, str) and resume_flag.lower() == "none":
        resume_flag = None
    start_epoch = 1
    if resume_flag:
        ckpt_path = Path(resume_flag) if isinstance(resume_flag, str) else utils.latest_checkpoint(output_dir)
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
                raw_inputs = batch["target"].to(device)
                inputs = model.image_to_model_range(raw_inputs)
                bs = raw_inputs.size(0)

                optimizer.zero_grad(set_to_none=True)
                if disc_optimizer:
                    disc_optimizer.zero_grad(set_to_none=True)

                try:
                    chunks = inputs.split(current_micro)
                    raw_chunks = raw_inputs.split(current_micro)
                    accum_steps = len(chunks)
                    d_loss_val = torch.tensor(0.0, device=device)
                    total_loss = torch.tensor(0.0, device=device)

                    for chunk, raw_chunk in zip(chunks, raw_chunks):
                        with autocast(device_type=device.type, enabled=use_amp):
                            if hasattr(model, "codebook"):
                                rec, vq_info = model(chunk)
                                vq_loss = vq_info["vq_loss"]
                                kl_term = torch.tensor(0.0, device=device)
                            else:
                                rec, posterior = model(chunk, sample_posterior=True)
                                vq_loss = torch.tensor(0.0, device=device)
                                kl_term = posterior.kl().mean()

                            rec_img = model.raw_output_to_image(rec, recon_type=recon_type)

                            if recon_type == "l1":
                                recon_loss = F.l1_loss(rec_img, raw_chunk)
                            elif recon_type == "mse":
                                recon_loss = F.mse_loss(rec_img, raw_chunk)
                            elif recon_type == "bce":
                                recon_loss = F.binary_cross_entropy_with_logits(rec, raw_chunk)
                            elif recon_type == "focal" or recon_type == "bce_focal":
                                recon_loss = bce_focal_loss(rec, raw_chunk, alpha=0.25, gamma=2.0, reduction="mean")
                            else:
                                raise ValueError(f"Unsupported recon_type '{recon_type}'.")

                            if perceptual is not None:
                                rec_p = rec_img if rec_img.device == perceptual_device else rec_img.to(perceptual_device)
                                chunk_p = raw_chunk if raw_chunk.device == perceptual_device else raw_chunk.to(perceptual_device)
                                perc_loss = perceptual(rec_p, chunk_p).to(device)
                            else:
                                perc_loss = torch.tensor(0.0, device=device)

                            disc_active = _disc_is_active(
                                discriminator=discriminator,
                                gan_weight=gan_weight,
                                gan_start=gan_start,
                                gan_start_steps=gan_start_steps,
                                epoch=epoch,
                                global_step=global_step,
                            )
                            if disc_active:
                                rec_d = rec_img if rec_img.device == disc_device else rec_img.to(disc_device)
                                chunk_d = raw_chunk if raw_chunk.device == disc_device else raw_chunk.to(disc_device)
                                fake_pred = discriminator(rec_d)
                                g_gan_loss = generator_hinge_loss(fake_pred).to(device)
                            else:
                                g_gan_loss = torch.tensor(0.0, device=device)

                            kl_scale = kl_weight
                            if kl_anneal_steps > 0:
                                step_for_anneal = max(1, global_step + 1)
                                kl_scale = kl_weight * min(1.0, step_for_anneal / max(1, kl_anneal_steps))

                            total_loss = (
                                recon_loss
                                + perceptual_weight * perc_loss
                                + kl_scale * kl_term
                                + effective_codebook_weight * vq_loss
                                + gan_weight * g_gan_loss
                            )

                        if scaler.is_enabled():
                            scaler.scale(total_loss / accum_steps).backward()
                        else:
                            (total_loss / accum_steps).backward()

                        if disc_active:
                            with autocast(device_type=disc_device.type, enabled=use_amp):
                                rec_detached = rec_img.detach()
                                raw_chunk_detached = raw_chunk.detach()
                                rec_d = rec_detached if rec_detached.device == disc_device else rec_detached.to(disc_device)
                                chunk_d = raw_chunk_detached if raw_chunk_detached.device == disc_device else raw_chunk_detached.to(disc_device)
                                real_pred = discriminator(chunk_d)
                                fake_pred_detached = discriminator(rec_d)
                                d_loss = discriminator_hinge_loss(real_pred, fake_pred_detached)
                            if scaler.is_enabled():
                                scaler.scale(d_loss / accum_steps).backward()
                            else:
                                (d_loss / accum_steps).backward()
                            d_loss_val = d_loss.detach()
                        else:
                            d_loss_val = torch.tensor(0.0, device=device)

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
                    raw_inputs = batch["target"].to(device)
                    inputs = model.image_to_model_range(raw_inputs)
                    chunks = inputs.split(min(current_micro, batch_size))
                    raw_chunks = raw_inputs.split(min(current_micro, batch_size))
                    for chunk, raw_chunk in zip(chunks, raw_chunks):
                        with autocast(device_type=device.type, enabled=use_amp):
                            if hasattr(model, "codebook"):
                                rec, vq_info = model(chunk)
                                vq_loss = vq_info["vq_loss"]
                                kl_term = torch.tensor(0.0, device=device)
                            else:
                                rec, posterior = model(chunk, sample_posterior=False)
                                vq_loss = torch.tensor(0.0, device=device)
                                kl_term = posterior.kl().mean()

                        rec_img = model.raw_output_to_image(rec, recon_type=recon_type)

                        if recon_type == "l1":
                            recon_loss = F.l1_loss(rec_img, raw_chunk)
                        elif recon_type == "mse":
                            recon_loss = F.mse_loss(rec_img, raw_chunk)
                        elif recon_type == "bce":
                            recon_loss = F.binary_cross_entropy_with_logits(rec, raw_chunk)
                        elif recon_type == "focal" or recon_type == "bce_focal":
                            recon_loss = bce_focal_loss(rec, raw_chunk, alpha=0.25, gamma=2.0, reduction="mean")
                        else:
                            raise ValueError(f"Unsupported recon_type '{recon_type}'.")

                        if perceptual is not None:
                            rec_p = rec_img if rec_img.device == perceptual_device else rec_img.to(perceptual_device)
                            chunk_p = raw_chunk if raw_chunk.device == perceptual_device else raw_chunk.to(perceptual_device)
                            perc_loss = perceptual(rec_p, chunk_p).to(device)
                        else:
                            perc_loss = torch.tensor(0.0, device=device)

                        disc_active = _disc_is_active(
                            discriminator=discriminator,
                            gan_weight=gan_weight,
                            gan_start=gan_start,
                            gan_start_steps=gan_start_steps,
                            epoch=epoch,
                            global_step=global_step,
                        )
                        if disc_active:
                            rec_d = rec_img if rec_img.device == disc_device else rec_img.to(disc_device)
                            chunk_d = raw_chunk if raw_chunk.device == disc_device else raw_chunk.to(disc_device)
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
                            recon_loss
                            + perceptual_weight * perc_loss
                            + kl_scale * kl_term
                            + effective_codebook_weight * vq_loss
                            + gan_weight * g_gan_loss
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

        # Always update last/best
        utils.save_checkpoint(state, output_dir / "vae_last.pt")
        if current_metric < best_metric:
            best_metric = current_metric
            state["best_metric"] = best_metric
            utils.save_checkpoint(state, output_dir / "vae_best.pt")
            logging.info("New best (%.6f) -> %s", best_metric, output_dir / "vae_best.pt")

        if utils.is_main_process():
            denom = max(num_samples, 1)
            metric_values = {
                "loss": totals["loss"] / denom,
                "recon": totals["recon"] / denom,
                "kl": totals["kl"] / denom,
                "vq": totals["vq"] / denom,
                "perceptual": totals["perceptual"] / denom,
                "g_gan": totals["g_gan"] / denom,
                "d_gan": totals["d_gan"] / denom,
            }
            row = [f"{epoch}"]
            for key in metrics_keys:
                value = metric_values.get(key)
                row.append("" if value is None else f"{value:.6f}")
            with metrics_path.open("a") as handle:
                handle.write(",".join(row) + "\n")

        # Periodic epoch artifacts
        should_save = epoch % save_every == 0 or epoch == epochs
        if should_save:
            epoch_dir = output_dir / "epochs" / f"epoch{epoch:04d}"
            ckpt_path = epoch_dir / "epoch.pt"
            utils.save_checkpoint(state, ckpt_path)
            logging.info("Saved epoch checkpoint: %s", ckpt_path)

            if visual_enabled and (epoch % visual_every == 0 or epoch == epochs):
                model.eval()
                with torch.no_grad():
                    sample_inputs = model.image_to_model_range(sample_batch)
                    with autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(sample_inputs, sample_posterior=False) if not hasattr(model, "codebook") else model(sample_inputs)
                    rec = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    rec_vis = model.raw_output_to_image(rec, recon_type=recon_type)
                    input_vis = sample_batch.clamp(0.0, 1.0)
                    input_grid = utils.make_grid(input_vis, 4, 5)
                    rec_grid = utils.make_grid(rec_vis, 4, 5)
                    noise = torch.randn((sample_count, *latent_shape_), device=device)
                    with autocast(device_type=device.type, enabled=use_amp):
                        gen = model.decode(noise)
                    gen_vis = model.raw_output_to_image(gen, recon_type=recon_type)
                    gen_grid = utils.make_grid(gen_vis, 4, 5)
                utils.save_image(input_grid, epoch_dir / "input.png")
                utils.save_image(rec_grid, epoch_dir / "recon.png")
                utils.save_image(gen_grid, epoch_dir / "gen.png")
                model.train()


def debug_visual_only(
    dataset,
    json_path: Path | str,
    ckpt_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    visual_samples: int = 10,
    seed: int | None = None,
) -> None:
    """
    Load a pretrained VAE checkpoint and save train-like visual outputs only.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    cfg = utils.load_json_config(json_path)
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("model_type", "")).lower()
    if model_type != "vae":
        raise ValueError(f"Expected model_type 'vae', got '{model_type}'.")
    training_cfg = cfg["training"]

    utils.set_seed(seed if seed is not None else training_cfg.get("seed"))
    device = utils.resolve_device(training_cfg.get("manual_device"), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_vae_model(cfg, device, ckpt_path=Path(ckpt_path), set_eval=True)
    recon_type = training_cfg.get("recon_type", "l1")

    out_root = Path(output_dir) if output_dir is not None else (Path(training_cfg.get("output_dir", "checkpoints/vae")) / "debug_train_like")
    out_root.mkdir(parents=True, exist_ok=True)

    indices = utils.select_visual_indices(dataset, int(visual_samples), seed=seed if seed is not None else training_cfg.get("seed"))
    batch = torch.stack([dataset[idx]["target"] for idx in indices], dim=0).to(device)
    with torch.no_grad():
        model_inputs = model.image_to_model_range(batch)
        outputs = model(model_inputs, sample_posterior=False)
        rec = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        rec_vis = model.raw_output_to_image(rec, recon_type=recon_type).clamp(0.0, 1.0)
    input_vis = batch.clamp(0.0, 1.0)

    rows = max(1, int(math.sqrt(rec_vis.size(0))))
    cols = max(1, rec_vis.size(0) // rows)
    utils.save_image(utils.make_grid(input_vis, rows, cols), out_root / "grid_input.png")
    utils.save_image(utils.make_grid(rec_vis, rows, cols), out_root / "grid_output.png")
    utils.save_image(utils.make_grid(input_vis, rows, cols), out_root / "grid_target.png")

    for b, idx in enumerate(indices):
        row = dataset.data[idx]
        save_output_tensor(dataset, row, dataset.target_key, input_vis[b].detach().cpu(), out_root / "target")
        save_output_tensor(dataset, row, dataset.target_key, rec_vis[b].detach().cpu(), out_root / "generated")
        if dataset.conditioning_key is not None and dataset[idx].get("image") is not None:
            save_output_tensor(dataset, row, dataset.conditioning_key, dataset[idx]["image"].detach().cpu(), out_root / "conditioning")

    logging.info("VAE debug visual-only generation completed for %d samples. Output: %s", len(indices), out_root)
    print(f"VAE debug visual-only generation completed for {len(indices)} samples.")
    print(f"Output directory: {out_root}")
