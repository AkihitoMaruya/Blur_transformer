#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:54:28 2026

@author: akihitomaruya
"""
# simmim_helpers/imagenet32_hf_loader.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as T

# HuggingFace datasets
from datasets import load_dataset

# Match SimMIM training defaults (timm constants)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Use the SAME MaskGenerator as training (you requested this)
from data.data_simmim_blur import MaskGenerator


class EnsureRGB:
    """Pickle-safe replacement for T.Lambda(lambda img: img.convert('RGB') ...)"""
    def __call__(self, img):
        return img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img


class _HFTorchWrapper(Dataset):
    """
    Wrap an HF Dataset so __getitem__ returns:
      img_tensor, label_int
    where img_tensor is float in [0,1] then optionally normalized by transform.
    """
    def __init__(self, hf_ds, transform: Optional[T.Compose] = None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        ex = self.hf_ds[int(idx)]
        img = ex["image"]         # PIL.Image
        label = int(ex["label"])  # int
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _AddMaskWrapper(Dataset):
    """
    Wrap a dataset that yields (img, label) into one that yields (img, mask, 0),
    using the SAME MaskGenerator logic as SimMIM training.
    """
    def __init__(self, base: Dataset, mask_generator: MaskGenerator):
        self.base = base
        self.mask_generator = mask_generator

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, _label = self.base[int(idx)]
        mask = self.mask_generator()              # np array (Hm,Wm) int {0,1}
        mask = torch.from_numpy(mask).float()     # float32 like training
        return img, mask, 0


def _infer_model_patch_size_from_config(config) -> int:
    if getattr(config.MODEL, "TYPE", None) == "swin":
        return int(config.MODEL.SWIN.PATCH_SIZE)
    if getattr(config.MODEL, "TYPE", None) == "vit":
        return int(config.MODEL.VIT.PATCH_SIZE)
    raise ValueError(f"Unsupported MODEL.TYPE in config: {getattr(config.MODEL, 'TYPE', None)!r}")


def load_imagenet32_hf(
    cache_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 2025,
    img_size: int = 32,
    transform: Optional[T.Compose] = None,
) -> Tuple[Dataset, Dataset, List[str], Dict[str, int]]:
    """
    Load benjamin-paine/imagenet-1k-32x32 (HF datasets), create deterministic train/val
    split (persisted), and return torch Datasets.

    Returns:
        train_ds, val_ds, class_names, class_to_idx
    """
    cwd = Path.cwd()

    # ---- cache directory (HF) ----
    if cache_dir is None:
        cache_dir = cwd / "data_set"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    # ---- where to save split indices ----
    if persist_dir is None:
        persist_dir = cwd / "Functions" / "_splits"
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    split_file = persist_dir / f"imagenet1k32_split_seed{seed}_val{int(val_ratio*100)}.pt"

    # ---- load HF dataset ----
    ds = load_dataset("benjamin-paine/imagenet-1k-32x32", cache_dir=str(cache_dir))

    # class names from HF features
    try:
        label_feature = ds["train"].features["label"]
        class_names: List[str] = list(label_feature.names)
    except Exception:
        seen = set(int(ex["label"]) for ex in ds["train"])
        max_label = max(seen)
        class_names = [f"class_{i}" for i in range(max_label + 1)]

    class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(class_names)}

    # ---- default transform if none provided ----
    if transform is None:
        tfms = [EnsureRGB()]
        if img_size != 32:
            tfms.append(T.Resize((img_size, img_size)))
        tfms.append(T.ToTensor())
        transform = T.Compose(tfms)

    hf_train = ds["train"]
    N = len(hf_train)

    if split_file.exists():
        idxs = torch.load(split_file, map_location="cpu")
        train_idx = idxs["train_idx"].long()
        val_idx = idxs["val_idx"].long()
        assert train_idx.numel() + val_idx.numel() == N, f"Split mismatch with dataset size {N}."
    else:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(N, generator=g)
        n_val = int(round(N * float(val_ratio)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        torch.save(
            {
                "train_idx": train_idx.cpu(),
                "val_idx": val_idx.cpu(),
                "seed": int(seed),
                "val_ratio": float(val_ratio),
                "num_samples": int(N),
                "class_to_idx": class_to_idx,
            },
            split_file,
        )

    base_ds = _HFTorchWrapper(hf_train, transform=transform)
    train_ds = Subset(base_ds, train_idx.tolist())
    val_ds = Subset(base_ds, val_idx.tolist())

    return train_ds, val_ds, class_names, class_to_idx


def build_imagenet32_loaders(
    *,
    cache_dir: str | Path,
    persist_dir: str | Path | None = None,
    # NEW: make config optional (fixes your error)
    config: Any | None = None,
    # If config is provided and return_masks=True -> yields (img, mask, 0)
    # Else -> yields (img, label)
    return_masks: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_ratio: float = 0.2,
    seed: int = 2025,
    img_size: int = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_STD),
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Return PyTorch DataLoaders for HF ImageNet32.

    Modes:
      - return_masks=False (default): yields (img, label)
      - return_masks=True + config provided: yields (img, mask, 0) using training MaskGenerator
    """
    tfms = [EnsureRGB()]
    if img_size != 32:
        tfms.append(T.Resize((img_size, img_size)))
    tfms.append(T.ToTensor())
    if normalize:
        tfms.append(T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    transform = T.Compose(tfms)

    train_ds, val_ds, class_names, class_to_idx = load_imagenet32_hf(
        cache_dir=cache_dir,
        persist_dir=persist_dir,
        val_ratio=float(val_ratio),
        seed=int(seed),
        img_size=int(img_size),
        transform=transform,
    )

    # Optional: add masks using SAME MaskGenerator as SimMIM training
    if bool(return_masks):
        if config is None:
            raise TypeError("build_imagenet32_loaders(return_masks=True) requires config=...")

        model_patch_size = _infer_model_patch_size_from_config(config)
        mask_gen = MaskGenerator(
            input_size=int(config.DATA.IMG_SIZE),
            mask_patch_size=int(config.DATA.MASK_PATCH_SIZE),
            model_patch_size=int(model_patch_size),
            mask_ratio=float(config.DATA.MASK_RATIO),
        )

        train_ds = _AddMaskWrapper(train_ds, mask_gen)
        val_ds = _AddMaskWrapper(val_ds, mask_gen)

    dl_train = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=True,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    return dl_train, dl_val, class_names, class_to_idx



