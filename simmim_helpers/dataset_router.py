#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:31:28 2025

@author: akihitomaruya
"""

# simmim_helpers/dataset_router.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data import build_loader  # ImageNet-style pretrain loader route


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _infer_train_val_roots_imagenet(data_path: str) -> tuple[str, str]:
    root = Path(data_path)
    train_root = root / "train"
    val_root   = root / "val"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Expected train dir: {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"Expected val dir: {val_root}")
    return str(train_root), str(val_root)


def build_loaders_pretrain(
    *,
    config,
    logger,
    dataset_mode: str,
    data_path: str,
    cifar_train_frac: float = 0.8,
    cifar_download: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Returns: (train_loader, val_loader)
      - CIFAR10: uses simmim_helpers/load_cifair10.py
      - ImageNet: uses SimMIM repo loaders (train via data.build_loader, val via data.data_simmim_blur.build_val_loader_simmim)
    """
    mode = str(dataset_mode).lower().strip()

    if mode == "cifar10":
        # IMPORTANT: local helper lives in simmim_helpers/ now
        from .load_cifar10 import build_cifar10_train_val_loaders



        train_loader, val_loader = build_cifar10_train_val_loaders(
            config=config,
            logger=logger,
            data_path=data_path,
            train_frac=float(cifar_train_frac),
            download=bool(cifar_download),
            is_distributed=_is_dist(),
        )
        logger.info(f"[CIFAR10] train_frac={float(cifar_train_frac)} download={bool(cifar_download)}")
        return train_loader, val_loader

    # ------------------------
    # ImageNet style (train/val folders)
    # ------------------------
    train_root, val_root = _infer_train_val_roots_imagenet(data_path)
    logger.info(f"[TRAIN] root = {train_root}")
    logger.info(f"[VAL]   root = {val_root}")

    # VAL loader
    from data.data_simmim_blur import build_val_loader_simmim
    data_loader_val = build_val_loader_simmim(config, logger, val_root)

    # TRAIN loader (use standard build_loader path)
    config.defrost()
    config.DATA.DATA_PATH = train_root
    config.freeze()
    data_loader_train = build_loader(config, logger, is_pretrain=True)

    return data_loader_train, data_loader_val

