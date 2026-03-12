#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:02:52 2025

@author: akihitomaruya
"""
# simmim_helpers/load_cifar10.py
# ---------------------------------------------------------
# CIFAR-10 loader for SimMIM that MATCHES data/data_simmim_blur.py mask logic:
#   - MaskGenerator(input_size, mask_patch_size, model_patch_size, mask_ratio)
#   - mask_count = ceil(token_count * mask_ratio)
#   - sample via np.random.permutation(token_count)[:mask_count]
#   - reshape (rand_size, rand_size)
#   - upsample by repeat(scale) to model-patch grid
# Returns batches like ImageNet SimMIM loaders:
#   (img, mask, 0)
# ---------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.247, 0.243, 0.261) 
# transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)



class MaskGenerator:
    """
    Exact copy of the MaskGenerator logic from data/data_simmim_blur.py
    (same math, same sampling, same upsample).
    """
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = int(input_size)
        self.mask_patch_size = int(mask_patch_size)
        self.model_patch_size = int(model_patch_size)
        self.mask_ratio = float(mask_ratio)

        assert self.input_size % self.mask_patch_size == 0, "IMG_SIZE must be divisible by MASK_PATCH_SIZE"
        assert self.mask_patch_size % self.model_patch_size == 0, "MASK_PATCH_SIZE must be divisible by model patch size"

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self) -> np.ndarray:
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask  # (Hm, Wm) at MODEL patch grid


class CIFAR10SimMIMDataset(Dataset):
    """
    Wrap a base dataset (e.g., Subset from random_split) so __getitem__ returns:
        img_norm, mask_float, 0
    and exposes `.config` so ValViz / helpers that assume dataset.config won't crash.
    """
    def __init__(self, base_ds, mask_generator: MaskGenerator, config):
        super().__init__()
        self.base_ds = base_ds
        self.mask_generator = mask_generator
        self.config = config  # ✅ for viz helpers expecting dataset.config

        # ALSO attach config onto the underlying dataset object when possible
        # (sometimes helpers do dataset.base_ds.dataset.config or similar)
        try:
            # Subset -> .dataset, plain dataset -> itself
            underlying = getattr(base_ds, "dataset", base_ds)
            setattr(underlying, "config", config)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        img, _ = self.base_ds[idx]  # img already transformed (tensor + normalize)

        mask = self.mask_generator()                    # numpy int
        mask = torch.from_numpy(mask).float()           # float32 like ImageNet SimMIM
        return img, mask, 0


def build_cifar10_train_val_loaders(
    config,
    logger,
    data_path: str,
    train_frac: float = 0.8,
    download: bool = True,
    is_distributed: bool = False,  # kept for API compatibility
):
    """
    Function expected by simmim_helpers/dataset_router.py.

    Returns:
        train_loader, val_loader
    where each batch yields (img, mask, 0).
    """

    # CIFAR10 is 32x32; keep transforms simple.
    # (If you want ImageNet-style aug later, change here only.)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        transform=transform,
        download=bool(download),
    )

    # Deterministic split
    n_total = len(full_train)
    n_train = int(round(float(train_frac) * n_total))
    n_val   = n_total - n_train
    g = torch.Generator().manual_seed(int(getattr(config, "SEED", 42)))
    train_base, val_base = random_split(full_train, [n_train, n_val], generator=g)

    # model_patch_size must match your model (same rule as data/data_simmim_blur.py)
    if config.MODEL.TYPE == "swin":
        model_patch_size = int(config.MODEL.SWIN.PATCH_SIZE)
    elif config.MODEL.TYPE == "vit":
        model_patch_size = int(config.MODEL.VIT.PATCH_SIZE)
    else:
        raise NotImplementedError(f"Unknown MODEL.TYPE: {config.MODEL.TYPE}")

    # MaskGenerator params must match SimMIM
    mask_gen = MaskGenerator(
        input_size=int(config.DATA.IMG_SIZE),
        mask_patch_size=int(config.DATA.MASK_PATCH_SIZE),
        model_patch_size=model_patch_size,
        mask_ratio=float(config.DATA.MASK_RATIO),
    )

    train_ds = CIFAR10SimMIMDataset(train_base, mask_gen, config=config)
    val_ds   = CIFAR10SimMIMDataset(val_base,   mask_gen, config=config)

    # MPS doesn't support pin_memory; only use it for CUDA
    pin = bool(getattr(config.DATA, "PIN_MEMORY", True)) and torch.cuda.is_available()
    workers = int(getattr(config.DATA, "NUM_WORKERS", 2))
    bs = int(getattr(config.DATA, "BATCH_SIZE", 64))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,          # OK because we are not using DistributedSampler on Mac
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
    )

    if logger is not None:
        logger.info(
            f"[CIFAR10 SimMIM] train={len(train_ds)} val={len(val_ds)} | "
            f"IMG_SIZE={int(config.DATA.IMG_SIZE)} MASK_PATCH_SIZE={int(config.DATA.MASK_PATCH_SIZE)} "
            f"MODEL_PATCH_SIZE={model_patch_size} MASK_RATIO={float(config.DATA.MASK_RATIO)} | "
            f"workers={workers} pin_memory={pin}"
        )

    return train_loader, val_loader
