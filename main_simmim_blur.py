#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 19:21:01 2025

@author: akihitomaruya
"""
# =========================================================
# main_simmim_blur.py  (FULL FILE, including val viz)
# =========================================================
# --------------------------------------------------------
# SimMIM pretrain main (BLUR / BLANK handled inside model)
# Modified by Akihito Maruya
#
# Clean refactor:
#   - dataset routing moved to simmim_helpers/dataset_router.py
#   - pos-noise & attention-span knobs moved to simmim_helpers/knobs.py
#   - viz helpers moved to simmim_helpers/viz_utils.py
#   - resume-safe epoch metrics + plotting in simmim_helpers/metrics_plot.py
#
# Plotting:
#   - logs train/val: rec/span/total into epoch_metrics.jsonl
#   - plots ONLY recon loss (train_rec=blue, val_rec=red) at end (rank0)
# --------------------------------------------------------

from __future__ import annotations

import os
import time
import argparse
import datetime
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
import wandb
from torch.amp import autocast, GradScaler

from data import build_loader, build_val_loader

from simmim_helpers.knobs import set_model_knobs, get_attention_span_loss
from simmim_helpers.viz_utils import (
    get_norm_stats,
    normalize,
    get_fixed_viz_indices,
    build_viz_loader_from_val_loader,
    save_triplet_grid,
)
from simmim_helpers.metrics_plot import append_metrics, plot_loss_curves

# OPTION A: canonical generator
from data.data_simmim_blur import MaskGenerator
from simmim_helpers.save_best import save_best_checkpoint


# ---------------------------------------------------------------------
# device + dist helpers
# ---------------------------------------------------------------------
def get_device() -> tuple[torch.device, bool]:
    """
    Priority: CUDA -> MPS -> CPU
    Returns (device, use_cuda_bool)
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_rank0() -> bool:
    return (not _is_dist()) or (_rank() == 0)


# ---------------------------------------------------------------------
# Mask generator wrapper
# ---------------------------------------------------------------------
def _infer_model_patch_size(config) -> int:
    if config.MODEL.TYPE == "swin":
        return int(config.MODEL.SWIN.PATCH_SIZE)
    if config.MODEL.TYPE == "vit":
        return int(config.MODEL.VIT.PATCH_SIZE)
    raise NotImplementedError(f"Unknown MODEL.TYPE: {config.MODEL.TYPE}")


def _make_mask_generator(config) -> MaskGenerator:
    model_patch = _infer_model_patch_size(config)
    return MaskGenerator(
        input_size=int(config.DATA.IMG_SIZE),
        mask_patch_size=int(config.DATA.MASK_PATCH_SIZE),
        model_patch_size=int(model_patch),
        mask_ratio=float(config.DATA.MASK_RATIO),
    )


def _unpack_batch(batch, *, mask_gen: MaskGenerator, device: torch.device, use_cuda: bool):
    """
    Robust across loaders.

    Accepts:
      - (img, mask)                 [SimMIMTransform]
      - (img, mask, anything)       [SimMIMValDataset]
      - (img, label)                [CIFAR10 / classification dataset]
    Returns:
      img:  Tensor [B,C,H,W] on device
      mask: Tensor [B,Hm,Wm] float32 on device
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    if len(batch) == 2:
        a, b = batch
        img = a
        if torch.is_tensor(b) and b.ndim >= 2:
            mask = b
        else:
            B = img.size(0)
            masks = [mask_gen() for _ in range(B)]
            mask = torch.from_numpy(np.stack(masks, axis=0)).float()
    elif len(batch) >= 3:
        img, mask = batch[0], batch[1]
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    img = img.to(device, non_blocking=use_cuda)

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device, non_blocking=use_cuda).float()

    return img, mask


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_option():
    parser = argparse.ArgumentParser("SimMIM pre-training script (BLUR/BLANK)", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding 'KEY VALUE' pairs.")

    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="dataset root (ImageNet: train/val; CIFAR10: download dir)",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--amp-opt-level", type=str, default="O0", choices=["O0", "O1", "O2"])
    parser.add_argument("--output", default="output", type=str, metavar="PATH")
    parser.add_argument("--tag", help="tag of experiment")

    # where to save figures (viz images + loss plots)
    parser.add_argument(
        "--figures-root",
        type=str,
        default=None,
        help="If set, save visualization figures under this directory (instead of <output>/viz_val_recon).",
    )

    # dataset selection
    parser.add_argument("--dataset", type=str, default=None, choices=["imagenet", "cifar10"])
    parser.add_argument("--cifar-download", action="store_true")
    parser.add_argument("--cifar-train-frac", type=float, default=0.8)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="disable Weights & Biases logging")
    parser.set_defaults(wandb=True)
    parser.add_argument("--wandb-project", type=str, default="vit-pretraining")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # corruption overrides
    parser.add_argument("--corruption", type=str, default=None, choices=["blur", "blank"])
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-patch-size", type=int, default=None)
    parser.add_argument("--blank-value", type=float, default=None)

    # blur params
    parser.add_argument("--blur-levels", type=str, default=None)
    parser.add_argument("--blur-height", type=int, default=None)
    parser.add_argument("--blur-order", type=int, default=None)
    parser.add_argument("--blur-seed", type=int, default=None)
    parser.add_argument("--blur-use-noise", action="store_true")
    parser.add_argument("--blur-noise-std", type=float, default=None)
    parser.add_argument("--blur-noise-seed", type=int, default=None)

    # val/viz
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--viz-freq", type=int, default=1)
    parser.add_argument("--val-only", action="store_true")

    # distributed
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    # loss mode
    parser.add_argument("--loss-on-full-image", action="store_true")

    # new knobs
    parser.add_argument("--lambda-span", type=float, default=0.0)
    parser.add_argument("--pos-noise-std", type=float, default=0.0)

    args = parser.parse_args()
    config = get_config(args)
    return args, config

def apply_main_overrides(args, config):
    config.defrost()

    # -------------------------
    # basic routing
    # -------------------------
    config.DATA.DATA_PATH = args.data_path

    if args.dataset is not None:
        config.DATA.DATASET = str(args.dataset).lower().strip()

    if args.corruption is not None:
        config.DATA.CORRUPTION = str(args.corruption).lower().strip()
    if args.mask_ratio is not None:
        config.DATA.MASK_RATIO = float(args.mask_ratio)
    if args.mask_patch_size is not None:
        config.DATA.MASK_PATCH_SIZE = int(args.mask_patch_size)
    if args.blank_value is not None:
        config.DATA.BLANK_VALUE = float(args.blank_value)

    # -------------------------
    # blur params
    # -------------------------
    if args.blur_levels is not None:
        config.DATA.BLUR_LEVELS = [
            s.strip() for s in str(args.blur_levels).split(",") if s.strip()
        ]
    if args.blur_height is not None:
        config.DATA.BLUR_HEIGHT = int(args.blur_height)
    if args.blur_order is not None:
        config.DATA.BLUR_ORDER = int(args.blur_order)
    if args.blur_seed is not None:
        config.DATA.BLUR_SEED = int(args.blur_seed)
    if args.blur_use_noise:
        config.DATA.BLUR_USE_NOISE = True
    if args.blur_noise_std is not None:
        config.DATA.BLUR_NOISE_STD = float(args.blur_noise_std)
    if args.blur_noise_seed is not None:
        config.DATA.BLUR_NOISE_SEED = int(args.blur_noise_seed)

    # -------------------------
    # AMP / optimizer knobs
    # -------------------------
    config.AMP_OPT_LEVEL = args.amp_opt_level

    if args.resume is not None:
        config.MODEL.RESUME = str(args.resume)

    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = int(args.batch_size)

    if args.accumulation_steps is not None:
        config.TRAIN.ACCUMULATION_STEPS = int(args.accumulation_steps)

   
    ds = str(config.DATA.DATASET).lower().strip()

    if ds == "cifar10":
        # CIFAR-10 stats (in [0,1] space)
        config.DATA.MEAN = (0.4914, 0.4822, 0.4465)
        config.DATA.STD  = (0.2470, 0.2430, 0.2610)
    elif ds == "imagenet":
        # ImageNet defaults
        config.DATA.MEAN = (0.485, 0.456, 0.406)
        config.DATA.STD  = (0.229, 0.224, 0.225)

    config.freeze()



# ---------------------------------------------------------------------
# Train / Val / Viz
# ---------------------------------------------------------------------
def train_one_epoch(
    config,
    model,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    logger,
    *,
    device,
    use_cuda,
    scaler=None,
) -> dict:
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    total_meter = AverageMeter()
    rec_meter = AverageMeter()
    span_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    accum = int(config.TRAIN.ACCUMULATION_STEPS) if int(config.TRAIN.ACCUMULATION_STEPS) > 1 else 1
    optimizer.zero_grad(set_to_none=True)

    mask_gen = _make_mask_generator(config)

    for idx, batch in enumerate(data_loader):
        img, mask = _unpack_batch(batch, mask_gen=mask_gen, device=device, use_cuda=use_cuda)

        use_amp = (use_cuda and config.AMP_OPT_LEVEL != "O0")

        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            rec_loss = model(img, mask)
            span_loss = get_attention_span_loss(model.module if hasattr(model, "module") else model)
            total_loss = rec_loss + span_loss

        if accum > 1:
            total_loss = total_loss / accum

        grad_norm = float("nan")

        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (idx + 1) % accum == 0:
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

        if use_cuda:
            torch.cuda.synchronize()

        # total_loss.item() is already divided by accum when accum>1, so multiply back for meters
        total_meter.update(float(total_loss.item()) * accum, img.size(0))
        rec_meter.update(float(rec_loss.item()), img.size(0))
        span_meter.update(float(span_loss.item()) if torch.is_tensor(span_loss) else float(span_loss), img.size(0))
        if isinstance(grad_norm, torch.Tensor):
            grad_norm_val = float(grad_norm.item())
        else:
            grad_norm_val = float(grad_norm)
        
        if np.isfinite(grad_norm_val):
            norm_meter.update(grad_norm_val)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and _is_rank0():
            lr = optimizer.param_groups[0]["lr"]
            mem = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if use_cuda else 0.0
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"total {total_meter.val:.4f} ({total_meter.avg:.4f})\t"
                f"rec {rec_meter.val:.4f} ({rec_meter.avg:.4f})\t"
                f"span {span_meter.val:.4f} ({span_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {mem:.0f}MB"
            )

    
    # If num_steps % accum != 0, do NOT do a final optimizer.step().
    # The leftover micro-batch gradients are discarded in the original script.

    if _is_rank0():
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(time.time() - start))}")

    return {
        "train_rec": float(rec_meter.avg),
        "train_span": float(span_meter.avg),
        "train_total": float(total_meter.avg),
    }


@torch.no_grad()
def validate_one_epoch(config, model, data_loader, epoch: int, *, logger, device, use_cuda) -> dict:
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    total_meter = AverageMeter()
    rec_meter = AverageMeter()
    span_meter = AverageMeter()

    start = time.time()
    end = time.time()

    mask_gen = _make_mask_generator(config)

    for idx, batch in enumerate(data_loader):
        img, mask = _unpack_batch(batch, mask_gen=mask_gen, device=device, use_cuda=use_cuda)

        use_amp = (use_cuda and config.AMP_OPT_LEVEL != "O0")

        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            rec_loss = model(img, mask)
            span_loss = get_attention_span_loss(model.module if hasattr(model, "module") else model)
            total_loss = rec_loss + span_loss

        total_meter.update(float(total_loss.item()), img.size(0))
        rec_meter.update(float(rec_loss.item()), img.size(0))
        span_meter.update(float(span_loss.item()) if torch.is_tensor(span_loss) else float(span_loss), img.size(0))

        if use_cuda:
            torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

        if _is_rank0() and (idx % config.PRINT_FREQ == 0):
            etas = batch_time.avg * (num_steps - idx)
            mem = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if use_cuda else 0.0
            logger.info(
                f"Val:   [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"total {total_meter.val:.4f} ({total_meter.avg:.4f})\t"
                f"rec {rec_meter.val:.4f} ({rec_meter.avg:.4f})\t"
                f"span {span_meter.val:.4f} ({span_meter.avg:.4f})\t"
                f"mem {mem:.0f}MB"
            )

    pack = torch.tensor([total_meter.avg, rec_meter.avg, span_meter.avg], device=device)
    if _is_dist():
        dist.all_reduce(pack, op=dist.ReduceOp.SUM)
        pack /= _world()

    if _is_rank0():
        logger.info(
            f"[Val] completed in {datetime.timedelta(seconds=int(time.time() - start))} | "
            f"avg total {pack[0].item():.6f} | avg rec {pack[1].item():.6f} | avg span {pack[2].item():.6f}"
        )

    return {
        "val_rec": float(pack[1].item()),
        "val_span": float(pack[2].item()),
        "val_total": float(pack[0].item()),
    }


@torch.no_grad()
def visualize_one_batch(model, viz_loader, out_path: Path, *, mean, std, device, use_cuda, config):
    """
    Expects viz_loader items to be (img, mask, ...) or (img, mask).
    Model returns (loss, x_rec01, x_clean01, x_corr01) when return_inputs01=True.
    """
    model.eval()
    batch = next(iter(viz_loader))

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        img = batch[0]
        mask = batch[1]
    else:
        raise RuntimeError("viz_loader must yield a tuple/list with at least (img, mask, ...).")

    img = img.to(device, non_blocking=use_cuda)
    mask = torch.as_tensor(mask, device=device).float()

    use_amp = (use_cuda and config.AMP_OPT_LEVEL != "O0")
    with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(img, mask, return_inputs01=True)
    if not (isinstance(out, (tuple, list)) and len(out) == 4):
        raise RuntimeError("Model must return (loss, x_rec01, x_clean01, x_corr01) when return_inputs01=True")

    _, x_rec01, _x_clean01, x_corr01 = out

    x_corr01 = x_corr01.float()
    x_rec01 = x_rec01.float()

    corr_norm = normalize(x_corr01, mean, std)
    rec_norm = normalize(x_rec01, mean, std)

    save_triplet_grid(img, corr_norm, rec_norm, out_path, mean=mean, std=std, n=25)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(config, args, logger, *, device, use_cuda):
    dataset_mode = str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()
    if args.dataset is not None:
        dataset_mode = str(args.dataset).lower().strip()
    logger.info(f"[DATASET] mode={dataset_mode}")
    if _is_rank0() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(config.OUTPUT),
            config=config,
        )

    mean, std = get_norm_stats(config, dataset_mode)

    # ------------------------------------------------------------
    # Use canonical data/ loaders (pretrain routes ImageNet vs CIFAR10)
    # ------------------------------------------------------------
    # Put CLI args into config so data/__init__.py can read them (defaults OK too)
    config.defrost()
    config.DATA.DATA_PATH = args.data_path
    # these keys are read by data/__init__.py in the patch I gave you
    config.DATA.CIFAR_TRAIN_FRAC = float(args.cifar_train_frac)
    config.DATA.CIFAR_DOWNLOAD = bool(args.cifar_download)
    config.freeze()
    
    data_loader_train = build_loader(config, logger, is_pretrain=True)
    
    # For ImageNet, pass val_path. For CIFAR10, val_path is ignored and split is used.
    val_path = None
    if dataset_mode != "cifar10":
        # assumes ImageNet layout: <data_path>/train and <data_path>/val
        val_path = os.path.join(args.data_path, "val")
    
    data_loader_val = build_val_loader(config, logger, val_path=val_path, is_pretrain=True)


    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)

    if hasattr(model, "loss_on_full_image"):
        model.loss_on_full_image = bool(args.loss_on_full_image)
        logger.info(f"[LOSS] loss_on_full_image={model.loss_on_full_image}")

    model = model.to(device)
    logger.info(str(model))
    

    model_without_ddp = model


    set_model_knobs(
        model_without_ddp,
        pos_noise_std=float(args.pos_noise_std),
        lambda_span=float(args.lambda_span),
        logger=logger,
    )
    if _is_rank0():
        enc = getattr(model_without_ddp, "encoder", model_without_ddp)
        logger.info(f"[SPAN DEBUG] encoder_file={enc.__class__.__module__} encoder_cls={enc.__class__.__name__}")

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

    out_dir = Path(config.OUTPUT)

    # -------------------------
    # Best checkpoint tracker (overwrite best.pt)
    # criterion: minimize val_rec
    # -------------------------
    best_metric = float("inf")
    best_epoch = -1
    best_metric_name = "val_rec"

    fig_root = Path(args.figures_root) if (getattr(args, "figures_root", None) and str(args.figures_root).strip()) else out_dir
    viz_dir = fig_root / "viz_val_recon"
    viz_dir.mkdir(parents=True, exist_ok=True)

    viz_loader = None
    if _is_rank0() and data_loader_val is not None:
        ds_len = len(data_loader_val.dataset)
        indices = get_fixed_viz_indices(ds_len, out_dir, seed=int(config.SEED), n=25)
        viz_loader = build_viz_loader_from_val_loader(data_loader_val, indices)
        logger.info(f"[ValViz] fixed indices file: {out_dir / 'viz_val_indices.npy'}")
        logger.info(f"[ValViz] saving images under: {viz_dir}")

    if args.val_only:
        logger.info("[Mode] VAL-ONLY: loading checkpoint then running one val + one visualization then exit.")
        if config.MODEL.RESUME:
            ckpt = torch.load(config.MODEL.RESUME, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(f"[VAL-ONLY] loaded: {config.MODEL.RESUME}")
            logger.info(f"[VAL-ONLY] load_state_dict: {msg}")

        val_stats = validate_one_epoch(config, model, data_loader_val, epoch=0, logger=logger, device=device, use_cuda=use_cuda)

        if _is_rank0() and viz_loader is not None:
            visualize_one_batch(
                model,
                viz_loader,
                viz_dir / "val_only_epoch_0000.png",
                mean=mean,
                std=std,
                device=device,
                use_cuda=use_cuda,
                config=config,
            )

        if _is_rank0():
            append_metrics(
                config.OUTPUT,
                {
                    "epoch": 0,
                    "run_ts": float(time.time()),
                    "lambda_span": float(args.lambda_span),
                    "pos_noise_std": float(args.pos_noise_std),
                    **val_stats,
                },
            )
            plot_loss_curves(config.OUTPUT, logger=logger)
        return

    optimizer = build_optimizer(config, model_without_ddp, logger, is_pretrain=True)

    if config.AMP_OPT_LEVEL != "O0" and not use_cuda:
        logger.warning("[AMP] Requested amp-opt-level != O0, but CUDA is not available. Running in FP32.")

    scaler = None
    if use_cuda and config.AMP_OPT_LEVEL != "O0":
        scaler = GradScaler("cuda")
    if _is_dist() and _world() > 1:
        if use_cuda:
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp,
                device_ids=[config.LOCAL_RANK],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        model_without_ddp = model.module
    else:
        model = model_without_ddp
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train, "sampler") and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        if data_loader_val is not None and hasattr(data_loader_val, "sampler") and hasattr(data_loader_val.sampler, "set_epoch"):
            data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            config,
            model,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            logger,
            device=device,
            use_cuda=use_cuda,
            scaler=scaler,
        )
        val_stats = {}
        if data_loader_val is not None and (epoch % int(args.val_freq) == 0):
            val_stats = validate_one_epoch(
                config, model, data_loader_val, epoch, logger=logger, device=device, use_cuda=use_cuda
            )

        # -------------------------
        # Save overwriting best.pt (rank0 only)
        # -------------------------
        if _is_rank0() and val_stats and (best_metric_name in val_stats):
            cur = float(val_stats[best_metric_name])
            if cur < best_metric:
                best_metric = cur
                best_epoch = int(epoch)

                save_best_checkpoint(
                    out_dir=out_dir,
                    epoch=epoch,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    args=args,
                    config=config,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    metric_name=best_metric_name,
                    scaler=scaler,
                )

                logger.info(f"[BEST] updated best.pt | epoch={best_epoch} {best_metric_name}={best_metric:.6f}")

        if _is_rank0():
            row = {
                "epoch": int(epoch),
                "run_ts": float(time.time()),
                "lambda_span": float(args.lambda_span),
                "pos_noise_std": float(args.pos_noise_std),
                **train_stats,
                **val_stats,
            }
            append_metrics(config.OUTPUT, row)
            if _is_rank0() and args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/rec": train_stats["train_rec"],
                    "train/span": train_stats["train_span"],
                    "train/total": train_stats["train_total"],
                    **({
                        "val/rec": val_stats["val_rec"],
                        "val/span": val_stats["val_span"],
                        "val/total": val_stats["val_total"],
                    } if val_stats else {}),
                })

        if data_loader_val is not None and _is_rank0() and (epoch % int(args.viz_freq) == 0):
            if viz_loader is not None:
                try:
                    visualize_one_batch(
                        model,
                        viz_loader,
                        viz_dir / f"val_epoch_{epoch:04d}.png",
                        mean=mean,
                        std=std,
                        device=device,
                        use_cuda=use_cuda,
                        config=config,
                    )
                    if args.wandb:
                        wandb.log({
                            "val/recon_grid": wandb.Image(str(viz_dir / f"val_epoch_{epoch:04d}.png")),
                            "epoch": epoch,
                        })
                except Exception as e:
                    logger.warning(f"[ValViz] failed at epoch {epoch}: {e}")

        if _is_rank0() and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger, scaler=scaler)

    if _is_rank0():
        plot_loss_curves(config.OUTPUT, logger=logger)

    total_time = time.time() - start_time
    logger.info("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    if _is_rank0() and args.wandb:
        wandb.finish()


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args, config = parse_option()
    apply_main_overrides(args, config)

    device, use_cuda = get_device()

    use_dist_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if use_dist_env:
        env_rank = int(os.environ["RANK"])
        env_world = int(os.environ["WORLD_SIZE"])
    else:
        env_rank = 0
        env_world = 1

    if use_dist_env and env_world > 1:
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(config.LOCAL_RANK)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=env_world,
            rank=env_rank,
        )
        dist.barrier()


    seed = int(config.SEED) + (_rank() if _is_dist() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_cuda:
        cudnn.benchmark = True

    # -----------------------------------------------------------------
    #  Match original SimMIM: linear LR scaling by global batch size
    # and (if used) accumulation steps.
    # -----------------------------------------------------------------
    global_batch = int(config.DATA.BATCH_SIZE) * int(_world())
    linear_scaled_lr = float(config.TRAIN.BASE_LR) * global_batch / 512.0
    linear_scaled_warmup_lr = float(config.TRAIN.WARMUP_LR) * global_batch / 512.0
    linear_scaled_min_lr = float(config.TRAIN.MIN_LR) * global_batch / 512.0

    if int(config.TRAIN.ACCUMULATION_STEPS) > 1:
        mult = int(config.TRAIN.ACCUMULATION_STEPS)
        linear_scaled_lr *= mult
        linear_scaled_warmup_lr *= mult
        linear_scaled_min_lr *= mult

    config.defrost()
    config.TRAIN.BASE_LR = float(linear_scaled_lr)
    config.TRAIN.WARMUP_LR = float(linear_scaled_warmup_lr)
    config.TRAIN.MIN_LR = float(linear_scaled_min_lr)
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=_rank(), name=f"{config.MODEL.NAME}")

    if _is_rank0():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(f"[Device] device={device} | cuda={use_cuda} | dist={_is_dist()} | world={_world()} | rank={_rank()}")
    logger.info(config.dump())

    main(config, args, logger, device=device, use_cuda=use_cuda)

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()
