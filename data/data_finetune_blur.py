#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 14:48:49 2026

@author: akihitomaruya
"""

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
#
# Modified: add CIFAR10 fine-tune support with minimal changes by Aki
# --------------------------------------------------------

from __future__ import annotations

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, create_transform

try:
    # older timm
    from timm.data.transforms import _pil_interp
except Exception:
    # newer timm
    from timm.data.transforms import str_to_pil_interp as _pil_interp


# CIFAR10 stats in [0,1]
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.247, 0.243, 0.261) 

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_world_rank():
    if _is_dist():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _cifar_split_indices(seed: int, train_frac: float, n_total: int = 50000):
    train_frac = float(train_frac)
    train_frac = min(max(train_frac, 0.05), 0.99)
    n_train = int(round(n_total * train_frac))

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx, val_idx


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)

    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    world, rank = _dist_world_rank()

    if _is_dist():
        sampler_train = DistributedSampler(dataset_train, num_replicas=world, rank=rank, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val,   num_replicas=world, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        sampler_train = None
        sampler_val = None
        shuffle_train = True

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        shuffle=shuffle_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=bool(getattr(config.DATA, "PIN_MEMORY", False)) and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(config.DATA.NUM_WORKERS > 0),
        prefetch_factor=4 if config.DATA.NUM_WORKERS > 0 else None,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        shuffle=False,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=bool(getattr(config.DATA, "PIN_MEMORY", False)) and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(config.DATA.NUM_WORKERS > 0),
        prefetch_factor=4 if config.DATA.NUM_WORKERS > 0 else None,
    )
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f"Fine-tune data transform, is_train={is_train}:\n{transform}")

    ds = str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()

    if ds == "imagenet":
        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
        return dataset, nb_classes

    if ds == "cifar10":
        root = config.DATA.DATA_PATH
        download = bool(getattr(config.DATA, "CIFAR_DOWNLOAD", False))

        # Default: official split (train=True vs train=False)
        use_split = bool(getattr(config.DATA, "CIFAR_FINETUNE_USE_SPLIT", False))
        if not use_split:
            dataset = CIFAR10(root=root, train=is_train, transform=transform, download=download)
            nb_classes = 10
            return dataset, nb_classes

        # Optional: split CIFAR10(train=True) into train/val
        base = CIFAR10(root=root, train=True, transform=transform, download=download)
        train_frac = float(getattr(config.DATA, "CIFAR_TRAIN_FRAC", 0.8))
        seed = int(getattr(config, "SEED", 42))
        train_idx, val_idx = _cifar_split_indices(seed=seed, train_frac=train_frac, n_total=len(base))
        idx = train_idx if is_train else val_idx
        dataset = Subset(base, idx)
        nb_classes = 10
        return dataset, nb_classes

    raise NotImplementedError(f"Unknown DATA.DATASET={ds}. Supported: imagenet, cifar10")


def build_transform(is_train, config):
    ds = str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()

    # keep upstream logic
    resize_im = config.DATA.IMG_SIZE > 32

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != "none" else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # upstream CIFAR-style tweak (RandomCrop instead of RandomResizedCrop)
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)

        # IMPORTANT: fix normalization for CIFAR10 when IMG_SIZE==32 path uses create_transform defaults
        if ds == "cifar10":
            # Replace final Normalize if present (timm compose usually ends with Normalize)
            for i in range(len(transform.transforms) - 1, -1, -1):
                if isinstance(transform.transforms[i], transforms.Normalize):
                    transform.transforms[i] = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)
                    break
        return transform

    # eval
    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())

    if ds == "cifar10":
        t.append(transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
