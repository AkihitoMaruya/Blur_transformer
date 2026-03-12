#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:14:46 2026

@author: akihitomaruya
"""

# simmim_helpers/msra_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class EnsureRGB:
    def __call__(self, img):
        return img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img


class MSRACroppedAll(Dataset):
    """
    root/
      images/*.jpg
      masks/*.png
    Returns (img, mask, 0) where mask is GT foreground/background.
    """
    def __init__(self, root: str | Path, transform: Optional[T.Compose] = None):
        self.root = Path(root).expanduser().resolve()
        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing: {self.img_dir}")
        if not self.msk_dir.exists():
            raise FileNotFoundError(f"Missing: {self.msk_dir}")

        img_paths = sorted(self.img_dir.glob("*.jpg"))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No .jpg found in {self.img_dir}")

        # keep only pairs
        self.img_paths = [p for p in img_paths if (self.msk_dir / f"{p.stem}.png").exists()]
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No image/mask pairs found under {self.root}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[int(idx)]
        mask_path = self.msk_dir / f"{img_path.stem}.png"

        img = Image.open(img_path).convert("RGB")
        m = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)

        m = np.array(m)
        m = (m > 0).astype(np.uint8)      # {0,1}
        m = torch.from_numpy(m).float()   # float like training masks

        return img, m, 0


def build_msra_loader(
    *,
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    img_size: int = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_DEFAULT_STD),
) -> DataLoader:
    tfms = [EnsureRGB()]
    if int(img_size) != 32:
        tfms.append(T.Resize((int(img_size), int(img_size))))
    tfms.append(T.ToTensor())
    if bool(normalize):
        tfms.append(T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    transform = T.Compose(tfms)

    ds = MSRACroppedAll(root, transform=transform)

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    return dl
