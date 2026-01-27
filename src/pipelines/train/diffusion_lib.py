"""
DDPM/score-based diffusion training loop adapted for the LDCT dataset.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils
from models.generators import DiffusionUNetFactory
from pipelines.utils import (
    build_scheduler,
    collect_conditioning_batch,
    resolve_conditioning_mode,
    sample_with_scheduler,
)
from utils import maybe_load_checkpoint, resolve_batch_size, resolve_device


def train(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    cfg = utils.load_json_config(json_path)
    if "diffusion" not in cfg:
        raise ValueError("Config does not declare a 'diffusion' section.")

    training_cfg = cfg["training"]
    model_cfg = cfg["diffusion"].get("unet", {})
    scheduler_cfg = cfg["diffusion"].get("scheduler", {})

    utils.setup_distributed(training_cfg.get("dist_backend"))
    utils.set_seed(training_cfg.get("seed"))
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = resolve_device(training_cfg.get("manual_device"), default_device)

    batch_size = resolve_batch_size(training_cfg, "train_batch_size", training_cfg.get("batch_size", 4))
    eval_batch_size = resolve_batch_size(training_cfg, "eval_batch_size", min(16, batch_size))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("num_epochs", training_cfg.get("epochs", 1)))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or cfg["diffusion"].get("conditioning")
    )
    channels = int(training_cfg.get("channels", model_cfg.get("out_channels", 1)))
    image_size = int(training_cfg.get("img_size", model_cfg.get("sample_size", 256)))
    save_image_epochs = int(training_cfg.get("save_image_epochs", training_cfg.get("save_every", 5)))
    save_model_epochs = int(training_cfg.get("save_model_epochs", training_cfg.get("save_every", 5)))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    lr_warmup = int(training_cfg.get("lr_warmup_steps", 500))

    base_output_dir = Path(training_cfg.get("output_dir", "checkpoints/diffusion"))
    output_dir = utils.allocate_run_dir(base_output_dir) if resume is None else base_output_dir
    training_cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "train_config.json"
    if not cfg_path.exists():
        utils.save_json_config(cfg_path, cfg)

    factory = DiffusionUNetFactory()
    model = factory.build(model_cfg, conditioning_mode, channels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler, num_inference_steps = build_scheduler(scheduler_cfg, training_cfg)
    num_train_steps = epochs * math.ceil(len(dataset) / batch_size)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=lr_warmup, num_training_steps=num_train_steps
    )

    train_sampler = DistributedSampler(dataset, shuffle=True) if utils.is_distributed() else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    use_amp = str(training_cfg.get("mixed_precision", "no")).lower() in {"fp16", "bf16", "true"}
    scaler = GradScaler(enabled=use_amp)

    eval_source = val_dataset if val_dataset is not None else dataset
    conditioning_eval = collect_conditioning_batch(eval_source, eval_batch_size, device) if utils.is_main_process() else None
    if conditioning_mode == "concatenate" and conditioning_eval is None:
        logging.warning("Diffusion config requested LDCT conditioning but dataset samples did not expose 'image'.")
    sample_count = conditioning_eval.size(0) if conditioning_eval is not None else eval_batch_size
    sample_shape = (sample_count, channels, image_size, image_size)

    resume_flag = Path(resume) if resume else None
    if resume_flag is None:
        resume_from_cfg = training_cfg.get("resume")
        if isinstance(resume_from_cfg, str) and resume_from_cfg.lower() != "none":
            resume_flag = Path(resume_from_cfg)
    start_epoch, best_metric = (
        maybe_load_checkpoint(resume_flag, "diffusion", model, optimizer, lr_scheduler, scaler)
        if resume_flag
        else (1, float("inf"))
    )

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        for batch in dataloader:
            clean = batch["target"].to(device, non_blocking=True)
            ldct = batch["image"]
            ldct = ldct.to(device, non_blocking=True) if ldct is not None else None

            bs = clean.size(0)
            chunk_size = max(1, math.ceil(bs / grad_accum))
            clean_chunks = clean.split(chunk_size)
            ldct_chunks = ldct.split(chunk_size) if ldct is not None else [None] * len(clean_chunks)

            optimizer.zero_grad(set_to_none=True)

            for clean_chunk, ldct_chunk in zip(clean_chunks, ldct_chunks):
                noise = torch.randn_like(clean_chunk)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (clean_chunk.size(0),), device=device
                ).long()
                noisy = scheduler.add_noise(clean_chunk, noise, timesteps)
                model_input = noisy
                if conditioning_mode == "concatenate" and ldct_chunk is not None:
                    model_input = torch.cat([noisy, ldct_chunk], dim=1)

                with autocast(device_type=device.type, enabled=use_amp):
                    pred = model(model_input, timesteps)
                    pred = pred[0] if isinstance(pred, (list, tuple)) else getattr(pred, "sample", pred)
                    loss = F.mse_loss(pred, noise)

                if scaler.is_enabled():
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()

                epoch_loss += loss.detach().item() * clean_chunk.size(0)
                num_samples += clean_chunk.size(0)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()

        loss_tensor = torch.tensor(epoch_loss, device=device)
        count_tensor = torch.tensor(num_samples, device=device)
        if utils.is_distributed():
            dist.all_reduce(loss_tensor)
            dist.all_reduce(count_tensor)
        avg_loss = (loss_tensor / count_tensor.clamp(min=1)).item()
        if utils.is_main_process():
            logging.info("Diffusion Epoch %03d | loss %.6f", epoch, avg_loss)

        current_metric = avg_loss
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "best_metric": best_metric,
        }

        if utils.is_main_process():
            utils.save_checkpoint(state, output_dir / "diff_last.pt")
            if current_metric < best_metric:
                best_metric = current_metric
                state["best_metric"] = best_metric
                utils.save_checkpoint(state, output_dir / "diff_best.pt")
                logging.info("New best diffusion loss %.6f -> %s", best_metric, output_dir / "diff_best.pt")

            if epoch % save_model_epochs == 0 or epoch == epochs:
                epoch_dir = output_dir / "epochs" / f"epoch{epoch:04d}"
                utils.save_checkpoint(state, epoch_dir / "epoch.pt")

        best_metric = min(best_metric, current_metric)

        cond_mode = conditioning_mode if conditioning_mode == "concatenate" and conditioning_eval is not None else None
        cond_batch = conditioning_eval if cond_mode else None
        save_samples = utils.is_main_process() and (epoch % save_image_epochs == 0 or epoch == epochs)
        if save_samples:
            model.eval()
            with torch.no_grad():
                samples = sample_with_scheduler(
                    model,
                    scheduler,
                    num_inference_steps,
                    sample_shape,
                    device,
                    conditioning_mode=cond_mode,
                    conditioning_batch=cond_batch,
                )
            vis = samples.clamp(0.0, 1.0)
            rows = max(1, int(math.sqrt(sample_shape[0])))
            cols = max(1, sample_shape[0] // rows)
            utils.save_image(utils.make_grid(vis, rows, cols), output_dir / "samples" / f"epoch{epoch:04d}.png")
            model.train()
