#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:32:49 2025

@author: akihitomaruya
"""

# simmim_helpers/viz_utils.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

# default stats
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_CIFAR10_MEAN  = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_CIFAR10_STD   = torch.tensor([0.2470, 0.243, 0.261]).view(1, 3, 1, 1)


def get_norm_stats(config, dataset: str):
    """
    Returns (mean, std) as (1,3,1,1) CPU tensors.
    Prefer config.DATA.MEAN/STD if present.
    """
    try:
        mean = getattr(config.DATA, "MEAN", None)
        std  = getattr(config.DATA, "STD", None)
        if mean is not None and std is not None:
            mean_t = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
            std_t  = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)
            return mean_t, std_t
    except Exception:
        pass

    ds = str(dataset).lower().strip()
    if ds == "cifar10":
        return _CIFAR10_MEAN, _CIFAR10_STD
    return _IMAGENET_MEAN, _IMAGENET_STD


def unnormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=x_norm.device, dtype=x_norm.dtype)
    std  = std.to(device=x_norm.device, dtype=x_norm.dtype)
    return x_norm * std + mean


def normalize(x01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=x01.device, dtype=x01.dtype)
    std  = std.to(device=x01.device, dtype=x01.dtype)
    return (x01 - mean) / std


def get_fixed_viz_indices(dataset_len: int, out_dir: Path, seed: int, n: int = 25) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "viz_val_indices.npy"

    if idx_path.is_file():
        idx = np.load(str(idx_path)).astype(np.int64)
        if idx.size >= n:
            return idx[:n]

    n = min(n, dataset_len)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(dataset_len, size=n, replace=False).astype(np.int64)
    np.save(str(idx_path), idx)
    return idx


def build_viz_loader_from_val_loader(data_loader_val, indices: np.ndarray) -> DataLoader:
    ds = data_loader_val.dataset
    subset = Subset(ds, indices.tolist())
    return DataLoader(
        subset,
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def save_triplet_grid(
    orig_norm: torch.Tensor,
    corr_norm: torch.Tensor,
    recon_norm: torch.Tensor,
    out_path: Path,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    n: int = 25,
):
    """
    Saves a 3-row grid: (orig, corrupted, recon) in pixel space [0,1].
    Inputs are normalized tensors.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, orig_norm.shape[0], corr_norm.shape[0], recon_norm.shape[0])

    orig  = unnormalize(orig_norm, mean, std).detach().cpu().float().clamp(0, 1)[:n]
    corr  = unnormalize(corr_norm, mean, std).detach().cpu().float().clamp(0, 1)[:n]
    recon = unnormalize(recon_norm, mean, std).detach().cpu().float().clamp(0, 1)[:n]

    nrow = 5
    g_o = make_grid(orig,  nrow=nrow, padding=2)
    g_c = make_grid(corr,  nrow=nrow, padding=2)
    g_r = make_grid(recon, nrow=nrow, padding=2)
    triplet = torch.cat([g_o, g_c, g_r], dim=2)
    save_image(triplet, str(out_path), normalize=False, value_range=(0, 1))
