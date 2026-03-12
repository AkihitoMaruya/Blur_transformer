#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 10:50:41 2025

@author: akihitomaruya
"""
# simmim_helpers/save_best.py
from __future__ import annotations

import os
from pathlib import Path
import torch


def _atomic_torch_save(obj, path: Path):
    """Write to tmp then replace (safer if job is interrupted)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def save_best_checkpoint(
    out_dir: Path,
    *,
    epoch: int,
    model_without_ddp,
    optimizer,
    lr_scheduler,
    args,
    config,
    best_metric: float,
    best_epoch: int,
    metric_name: str,
    scaler=None,
):
    """
    Save a single overwriting best.pt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "metric_name": str(metric_name),
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": {
            "lambda_span": float(args.lambda_span),
            "pos_noise_std": float(args.pos_noise_std),
        },
        "config": config.dump(),
    }

    _atomic_torch_save(ckpt, out_dir / "best.pt")