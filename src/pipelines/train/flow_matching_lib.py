"""
Flow-matching training loop adapted for the LDCT dataset.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils
from utils.model_utils.diffusion_utils import build_diffusion_model, prepare_diffusion_visual_batch, decode_diffusion_batch
from pipelines.utils import build_scheduler, resolve_conditioning_mode
from utils import maybe_load_checkpoint, resolve_batch_size, resolve_device


def train(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    cfg = utils.load_json_config(json_path)
    if "model" not in cfg:
        raise ValueError("Config does not declare a 'model' section.")
    model_block = cfg["model"]
    model_type = str(model_block.get("model_type", "")).lower()
    if model_type != "flow_matching":
        raise ValueError(f"Expected model_type 'flow_matching', got '{model_type}'.")

    training_cfg = cfg["training"]
    scheduler_cfg = model_block.get("scheduler", {})

    utils.setup_distributed(training_cfg.get("dist_backend"))
    utils.set_seed(training_cfg.get("seed"))
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = resolve_device(training_cfg.get("manual_device"), default_device)

    batch_size = resolve_batch_size(training_cfg, "train_batch_size", training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("num_epochs", training_cfg.get("epochs", 1)))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_block.get("conditioning")
    )
    save_model_epochs = int(training_cfg.get("save_model_epochs", training_cfg.get("save_every", 5)))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    lr_warmup = int(training_cfg.get("lr_warmup_steps", 500))

    base_output_dir = Path(training_cfg.get("output_dir", "checkpoints/flow_matching"))
    output_dir = utils.allocate_run_dir(base_output_dir) if resume is None else base_output_dir
    training_cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "train_config.json"
    if not cfg_path.exists():
        utils.save_json_config(cfg_path, cfg)

    model = build_diffusion_model(cfg, device, ckpt_path=None, set_eval=False)
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

    visual_enabled = bool(training_cfg.get("save_images", False))
    visual_every = int(training_cfg.get("save_images_every", 10))
    visual_targets = None
    visual_cond = None
    if visual_enabled and utils.is_main_process():
        eval_source = val_dataset if val_dataset is not None else dataset
        visual_count = int(training_cfg.get("visual_samples", 8))
        visual_targets, visual_cond = prepare_diffusion_visual_batch(eval_source, visual_count, device)
        if conditioning_mode == "concatenate" and visual_cond is None:
            logging.warning("Flow matching config requested conditioning but dataset samples did not expose 'image'.")

    metrics_path = output_dir / "metrics.csv"
    if utils.is_main_process() and not metrics_path.exists():
        header = "epoch,train_loss\n"
        metrics_path.write_text(header)

    resume_flag = Path(resume) if resume else None
    if resume_flag is None:
        resume_from_cfg = training_cfg.get("resume")
        if isinstance(resume_from_cfg, str) and resume_from_cfg.lower() != "none":
            resume_flag = Path(resume_from_cfg)
    start_epoch, best_metric = (
        maybe_load_checkpoint(resume_flag, "flow", model, optimizer, lr_scheduler, scaler) if resume_flag else (1, float("inf"))
    )

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        train_loop = tqdm(
            dataloader,
            desc=f"Train {epoch}/{epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not utils.is_main_process(),
        )
        for batch in train_loop:
            clean = batch["target"].to(device, non_blocking=True)
            ldct = batch["image"]
            ldct = ldct.to(device, non_blocking=True) if ldct is not None else None

            bs = clean.size(0)
            chunk_size = max(1, math.ceil(bs / grad_accum))
            clean_chunks = clean.split(chunk_size)
            ldct_chunks = ldct.split(chunk_size) if ldct is not None else [None] * len(clean_chunks)

            optimizer.zero_grad(set_to_none=True)

            for c_idx, (clean_chunk, ldct_chunk) in enumerate(zip(clean_chunks, ldct_chunks)):
                noise = torch.randn_like(clean_chunk)
                t = torch.rand(clean_chunk.size(0), device=device)
                timesteps = (t * (scheduler.config.num_train_timesteps - 1)).long()
                x_t = (1.0 - t[:, None, None, None]) * clean_chunk + t[:, None, None, None] * noise
                model_input = x_t
                if conditioning_mode == "concatenate" and ldct_chunk is not None:
                    model_input = torch.cat([x_t, ldct_chunk], dim=1)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(model_input, timesteps)
                    pred = pred[0] if isinstance(pred, (list, tuple)) else getattr(pred, "sample", pred)
                    target = noise - clean_chunk
                    loss = F.mse_loss(pred, target)

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
            if utils.is_main_process():
                denom = max(num_samples, 1)
                train_loop.set_postfix(loss=epoch_loss / denom)

        loss_tensor = torch.tensor(epoch_loss, device=device)
        count_tensor = torch.tensor(num_samples, device=device)
        if utils.is_distributed():
            dist.all_reduce(loss_tensor)
            dist.all_reduce(count_tensor)
        avg_loss = (loss_tensor / count_tensor.clamp(min=1)).item()
        if utils.is_main_process():
            logging.info("FlowMatch Epoch %03d | loss %.6f", epoch, avg_loss)

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
            utils.save_checkpoint(state, output_dir / "flow_last.pt")
            if current_metric < best_metric:
                best_metric = current_metric
                state["best_metric"] = best_metric
                utils.save_checkpoint(state, output_dir / "flow_best.pt")
                logging.info("New best flow-matching loss %.6f -> %s", best_metric, output_dir / "flow_best.pt")

            if epoch % save_model_epochs == 0 or epoch == epochs:
                epoch_dir = output_dir / "epochs" / f"epoch{epoch:04d}"
                utils.save_checkpoint(state, epoch_dir / "epoch.pt")

        best_metric = min(best_metric, current_metric)

        save_samples = (
            visual_enabled
            and utils.is_main_process()
            and visual_targets is not None
            and (epoch % visual_every == 0 or epoch == epochs)
        )
        if save_samples:
            model.eval()
            with torch.no_grad():
                outputs = decode_diffusion_batch(
                    model,
                    training_cfg,
                    cfg["model"],
                    device,
                    visual_targets.shape,
                    visual_cond if conditioning_mode == "concatenate" else None,
                )
            vis = outputs.clamp(0.0, 1.0)
            input_vis = visual_cond if visual_cond is not None else visual_targets
            rows = max(1, int(math.sqrt(vis.size(0))))
            cols = max(1, vis.size(0) // rows)
            utils.save_image(utils.make_grid(input_vis, rows, cols), output_dir / "visuals" / f"epoch{epoch:04d}_input.png")
            utils.save_image(utils.make_grid(vis, rows, cols), output_dir / "visuals" / f"epoch{epoch:04d}_output.png")
            utils.save_image(utils.make_grid(visual_targets, rows, cols), output_dir / "visuals" / f"epoch{epoch:04d}_target.png")
            model.train()

        if utils.is_main_process():
            with metrics_path.open("a") as handle:
                handle.write(f"{epoch},{avg_loss:.6f}\n")
