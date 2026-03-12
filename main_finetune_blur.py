#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:26:13 2025

@author: akihitomaruya
"""
#!/usr/bin/env python3
# =========================================================
# main_finetune_blur.py
#
# Fine-tune SimMIM encoders (ViT/Swin) for classification.
#
# - Keeps structure close to upstream SimMIM fine-tune script.
# - Uses repo loader:
#       dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
#           = build_loader(config, logger, is_pretrain=False)
# - NO visualization code.
# - Optional knobs: --lambda-span, --pos-noise-std
# - Cosine scheduler compat for timm>=1.0 WITHOUT editing lr_scheduler.py
# - DDP only when torchrun env present AND CUDA.
#
# UPDATES YOU REQUESTED:
#    Show running train accuracy ALWAYS (even with mixup/cutmix) using proxy hard labels (argmax).
#    Show running val accuracy (already true) + keep epoch-level train/val acc history.
#    Save epoch metrics to <OUTPUT>/metrics.jsonl (train/val acc1/acc5 + losses + lr).
# =========================================================

from __future__ import annotations

import os
import time
import argparse
import datetime
import json
import numpy as np
from typing import Dict, Any, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_val_loader  # build_val_loader not used for finetune (kept per request)
from lr_scheduler import build_scheduler  # keep import; no file edits
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint,
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
)

from torch.amp import autocast, GradScaler
import wandb

from simmim_helpers.knobs import set_model_knobs, get_attention_span_loss


# -----------------------------------------------------------------------------
# dist helpers (safe on non-ddp)
# -----------------------------------------------------------------------------
def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_rank0() -> bool:
    return (not _is_dist()) or (_rank() == 0)


# -----------------------------------------------------------------------------
# device helper
# -----------------------------------------------------------------------------
def _select_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


# -----------------------------------------------------------------------------
# metrics logging
# -----------------------------------------------------------------------------
def _metrics_path(config) -> str:
    return os.path.join(config.OUTPUT, "metrics.jsonl")


def _append_metrics(config, row: Dict[str, Any]) -> None:
    if not _is_rank0():
        return
    path = _metrics_path(config)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# -----------------------------------------------------------------------------
# Scheduler compat (NO lr_scheduler.py edits)
# -----------------------------------------------------------------------------
def build_scheduler_compat(config, optimizer, n_iter_per_epoch: int):
    """
    If config requests cosine, build a timm CosineLRScheduler with a signature
    compatible with timm>=1.0. Otherwise fall back to repo build_scheduler().
    """
    name = str(getattr(config.TRAIN.LR_SCHEDULER, "NAME", "cosine")).lower().strip()
    if name != "cosine":
        return build_scheduler(config, optimizer, n_iter_per_epoch)

    from timm.scheduler.cosine_lr import CosineLRScheduler
    import inspect

    epochs = int(config.TRAIN.EPOCHS)
    warmup_epochs = int(getattr(config.TRAIN, "WARMUP_EPOCHS", 0))

    t_initial = epochs * int(n_iter_per_epoch)
    warmup_t = warmup_epochs * int(n_iter_per_epoch)

    sig = inspect.signature(CosineLRScheduler.__init__).parameters

    kwargs = dict(
        optimizer=optimizer,
        t_initial=t_initial,
        lr_min=float(config.TRAIN.MIN_LR),
        warmup_lr_init=float(config.TRAIN.WARMUP_LR),
        warmup_t=warmup_t,
        warmup_prefix=True,
        t_in_epochs=False,
        cycle_limit=1,
    )

    # timm signature differences
    if "cycle_mul" in sig:
        kwargs["cycle_mul"] = 1.0
    elif "t_mul" in sig:
        kwargs["t_mul"] = 1.0

    if "cycle_decay" in sig:
        kwargs["cycle_decay"] = 1.0

    return CosineLRScheduler(**kwargs)


# -----------------------------------------------------------------------------
# CLI (close to upstream; local_rank optional for non-DDP)
# -----------------------------------------------------------------------------
def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding 'KEY VALUE' pairs.")

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--pretrained", type=str, help="path to pre-trained model")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="disable Weights & Biases logging")
    parser.set_defaults(wandb=True)
    parser.add_argument("--wandb-project", type=str, default="vit-finetune")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # distributed training (optional; default from env)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="local rank for DistributedDataParallel (torchrun)",
    )

    # knobs
    parser.add_argument("--lambda-span", type=float, default=0.0)
    parser.add_argument("--pos-noise-std", type=float, default=0.0)

    args = parser.parse_args()
    config = get_config(args)
    return args, config

def apply_main_overrides(args, config):
    config.defrost()

    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
    if args.resume is not None:
        config.MODEL.RESUME = str(args.resume)
    if args.pretrained is not None:
        config.PRETRAINED = str(args.pretrained)
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = int(args.batch_size)
    if args.accumulation_steps is not None:
        config.TRAIN.ACCUMULATION_STEPS = int(args.accumulation_steps)

    config.AMP_OPT_LEVEL = str(args.amp_opt_level)

    # =========================================================
    # Sync normalization with dataset (finetune too)
    # =========================================================
    ds = str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()

    if ds == "cifar10":
        # CIFAR-10 stats (in [0,1] space)
        config.DATA.MEAN = (0.4914, 0.4822, 0.4465)
        config.DATA.STD  = (0.2470, 0.2430, 0.2610)
    elif ds == "imagenet":
        # ImageNet defaults
        config.DATA.MEAN = (0.485, 0.456, 0.406)
        config.DATA.STD  = (0.229, 0.224, 0.225)

    config.freeze()

# -----------------------------------------------------------------------------
# Train / Val
# -----------------------------------------------------------------------------
def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    device,
    logger,
    scaler=None,
):
    model.train()
    optimizer.zero_grad()

    if _is_rank0():
        logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    ce_meter = AverageMeter()
    span_meter = AverageMeter()

    # running accuracy meters (ALWAYS numeric, proxy if mixup/cutmix)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()

    # note shown in logs when we are using proxy targets
    using_proxy_any = False

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))

        mixed = False
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            mixed = True

        use_amp = (device.type == "cuda" and config.AMP_OPT_LEVEL != "O0")

        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(samples)
            span_loss = get_attention_span_loss(model.module if hasattr(model, "module") else model)
            ce_loss = criterion(outputs, targets)


        #  ALWAYS compute accuracy:
        # if targets are soft (B,C), use argmax as proxy hard labels
        if targets.ndim == 2:
            hard_targets = targets.argmax(dim=1)
            using_proxy = True
            using_proxy_any = True
        else:
            hard_targets = targets
            using_proxy = False

        acc1, acc5 = accuracy(outputs, hard_targets, topk=(1, 5))
        if _is_dist():
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

        # backward / step
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = (ce_loss + span_loss) / config.TRAIN.ACCUMULATION_STEPS
        else:
            loss = (ce_loss + span_loss)
        
        grad_norm = float("nan")
        
        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        do_step = (
            config.TRAIN.ACCUMULATION_STEPS == 1
            or ((idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        )
        
        if do_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
        
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
        
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step_update(epoch * num_steps + idx)

        

        if device.type == "cuda":
            torch.cuda.synchronize()

        # meters
        bs = int(samples.size(0))
        loss_meter.update(float(loss.item()), bs)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm_val = float(grad_norm.item())
        else:
            grad_norm_val = float(grad_norm)
        
        if np.isfinite(grad_norm_val):
            norm_meter.update(grad_norm_val)
        ce_meter.update(float(ce_loss.detach().item()), bs)
        span_meter.update(float(span_loss.detach().item()) if torch.is_tensor(span_loss) else float(span_loss), bs)

        acc1_meter.update(float(acc1.item()), bs)
        acc5_meter.update(float(acc5.item()), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and _is_rank0():
            lr = optimizer.param_groups[-1]["lr"]
            memory_used = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
            etas = batch_time.avg * (num_steps - idx)

            proxy_note = " (proxy: soft->argmax)" if (mixed and using_proxy) else ""

            # Match your desired style closely
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB\t"
                f"ce {ce_meter.val:.4f} ({ce_meter.avg:.4f})\t"
                f"span {span_meter.val:.4f} ({span_meter.avg:.4f})\t"
                f"acc1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})\t"
                f"acc5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})"
                f"{proxy_note}"
            )

    epoch_time = time.time() - start
    if _is_rank0():
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    # epoch summary (return averages)
    return {
        "train_loss": float(loss_meter.avg),
        "train_ce": float(ce_meter.avg),
        "train_span": float(span_meter.avg),
        "train_acc1": float(acc1_meter.avg),
        "train_acc5": float(acc5_meter.avg),
        "train_used_proxy": bool(using_proxy_any),
    }


@torch.no_grad()
def validate(config, data_loader, model, device, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    ce_meter = AverageMeter()
    span_meter = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(data_loader):
        # common: (images, target)
        images, target = batch[0], batch[1]
        images = images.to(device, non_blocking=(device.type == "cuda"))
        target = target.to(device, non_blocking=(device.type == "cuda"))

        use_amp = (device.type == "cuda" and config.AMP_OPT_LEVEL != "O0")

        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(images)
            ce_loss = criterion(output, target)
            span_loss = get_attention_span_loss(model.module if hasattr(model, "module") else model)
            loss = ce_loss + span_loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if _is_dist():
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
            ce_loss = reduce_tensor(ce_loss)
            if torch.is_tensor(span_loss):
                span_loss = reduce_tensor(span_loss)
            else:
                span_loss = reduce_tensor(torch.tensor(float(span_loss), device=output.device))

        bs = int(target.size(0))
        loss_meter.update(float(loss.item()), bs)
        ce_meter.update(float(ce_loss.item()), bs)
        span_meter.update(float(span_loss.item()) if torch.is_tensor(span_loss) else float(span_loss), bs)
        acc1_meter.update(float(acc1.item()), bs)
        acc5_meter.update(float(acc5.item()), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and _is_rank0():
            memory_used = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"CE {ce_meter.val:.4f} ({ce_meter.avg:.4f})\t"
                f"Span {span_meter.val:.4f} ({span_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )

    if _is_rank0():
        logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")

    return {
        "val_loss": float(loss_meter.avg),
        "val_ce": float(ce_meter.avg),
        "val_span": float(span_meter.avg),
        "val_acc1": float(acc1_meter.avg),
        "val_acc5": float(acc5_meter.avg),
    }


@torch.no_grad()
def throughput(data_loader, model, logger, device):
    model.eval()
    for _, (images, _) in enumerate(data_loader):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        batch_size = images.shape[0]
        for _ in range(50):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("throughput averaged with 30 times")
        tic1 = time.time()
        for _ in range(30):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(config, logger, args, device, use_cuda: bool):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False).to(device)
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if _is_rank0() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(config.OUTPUT),
            config=config,
        )

    # AMP (apex) CUDA-only
    if config.AMP_OPT_LEVEL != "O0" and not use_cuda:
        if _is_rank0():
            logger.warning("[AMP] AMP requested but CUDA is not available. Forcing AMP_OPT_LEVEL=O0.")
        config.defrost()
        config.AMP_OPT_LEVEL = "O0"
        config.freeze()
    
    scaler = None
    if use_cuda and config.AMP_OPT_LEVEL != "O0":
        scaler = GradScaler("cuda")

    # DDP only if dist initialized AND CUDA
    if _is_dist():
        if device.type != "cuda":
            raise RuntimeError("DDP enabled but device is not CUDA. Run single-process on MPS/CPU.")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # knobs
    set_model_knobs(
        model_without_ddp,
        pos_noise_std=float(args.pos_noise_std),
        lambda_span=float(args.lambda_span),
        logger=logger,
    )
    if _is_rank0():
        logger.info(f"[KNOBS] lambda_span={float(args.lambda_span)} | pos_noise_std={float(args.pos_noise_std)}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler_compat(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    # resume / pretrained
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

        # re-apply knobs after load
        set_model_knobs(
            model_without_ddp,
            pos_noise_std=float(args.pos_noise_std),
            lambda_span=float(args.lambda_span),
            logger=logger,
        )

        val_row = validate(config, data_loader_val, model, device, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {val_row['val_acc1']:.1f}%")
        if config.EVAL_MODE or args.eval:
            return

    elif getattr(config, "PRETRAINED", None):
        load_pretrained(config, model_without_ddp, logger)
        set_model_knobs(
            model_without_ddp,
            pos_noise_std=float(args.pos_noise_std),
            lambda_span=float(args.lambda_span),
            logger=logger,
        )

    if config.THROUGHPUT_MODE or args.throughput:
        throughput(data_loader_val, model, logger, device)
        return

    if args.eval or config.EVAL_MODE:
        val_row = validate(config, data_loader_val, model, device, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {val_row['val_acc1']:.1f}%")
        return

    # history containers (epoch-level)
    history = {
        "train_acc1": [],
        "train_acc5": [],
        "val_acc1": [],
        "val_acc5": [],
        "train_loss": [],
        "val_loss": [],
    }

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train, "sampler") and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_row = train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            device,
            logger,
            scaler=scaler,
        )
        val_row = validate(config, data_loader_val, model, device, logger)
        

        # store epoch history
        history["train_acc1"].append(train_row["train_acc1"])
        history["train_acc5"].append(train_row["train_acc5"])
        history["val_acc1"].append(val_row["val_acc1"])
        history["val_acc5"].append(val_row["val_acc5"])
        history["train_loss"].append(train_row["train_loss"])
        history["val_loss"].append(val_row["val_loss"])

        # save checkpoint
        if _is_rank0() and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
           save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, scaler=scaler)
        # epoch summary logs
        if _is_rank0():
            logger.info(
                f"[EPOCH {epoch}] "
                f"train_acc1={train_row['train_acc1']:.3f} train_acc5={train_row['train_acc5']:.3f} "
                f"val_acc1={val_row['val_acc1']:.3f} val_acc5={val_row['val_acc5']:.3f} "
                f"train_loss={train_row['train_loss']:.4f} val_loss={val_row['val_loss']:.4f}"
            )

        # track best val acc1
        max_accuracy = max(max_accuracy, val_row["val_acc1"])
        if _is_rank0():
            logger.info(f"Max val acc1 so far: {max_accuracy:.2f}%")
        if _is_rank0() and args.wandb:
            wandb.log({
                "epoch": epoch,
                "lr": float(optimizer.param_groups[-1]["lr"]),
                "train/loss": train_row["train_loss"],
                "train/ce": train_row["train_ce"],
                "train/span": train_row["train_span"],
                "train/acc1": train_row["train_acc1"],
                "train/acc5": train_row["train_acc5"],
                "val/loss": val_row["val_loss"],
                "val/ce": val_row["val_ce"],
                "val/span": val_row["val_span"],
                "val/acc1": val_row["val_acc1"],
                "val/acc5": val_row["val_acc5"],
                "max_val_acc1": float(max_accuracy),
            })
        # write epoch metrics (jsonl)
        _append_metrics(
            config,
            dict(
                epoch=int(epoch),
                lr=float(optimizer.param_groups[-1]["lr"]),
                **train_row,
                **val_row,
                max_val_acc1=float(max_accuracy),
            ),
        )

    # save a simple history json (optional, easy plotting)
    if _is_rank0():
        with open(os.path.join(config.OUTPUT, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    logger.info("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    if _is_rank0() and args.wandb:
        wandb.finish()


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args, config = parse_option()
    apply_main_overrides(args, config)

    device, use_cuda = _select_device()

    # Dist init: if torchrun env exists, init process group even for WORLD_SIZE=1
    use_dist_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if use_dist_env:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(getattr(args, "local_rank", int(os.environ.get("LOCAL_RANK", 0))))
    else:
        rank = 0
        world_size = 1
    
    if use_dist_env and world_size > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
    
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
    else:
        rank = 0
        world_size = 1


    # AMP only if CUDA
    if config.AMP_OPT_LEVEL != "O0" and not use_cuda:
        config.defrost()
        config.AMP_OPT_LEVEL = "O0"
        config.freeze()

    
    seed = int(config.SEED) + (_rank() if _is_dist() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = (device.type == "cuda")

    # linear scale LR like upstream
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * _world() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * _world() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * _world() / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = float(linear_scaled_lr)
    config.TRAIN.WARMUP_LR = float(linear_scaled_warmup_lr)
    config.TRAIN.MIN_LR = float(linear_scaled_min_lr)
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=_rank(), name=f"{config.MODEL.NAME}")

    if _is_rank0():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(f"[Device] device={device} | cuda={use_cuda} | dist={_is_dist()} | world={_world()} | rank={_rank()}")
    logger.info(config.dump())

    main(config, logger, args, device=device, use_cuda=use_cuda)

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()
