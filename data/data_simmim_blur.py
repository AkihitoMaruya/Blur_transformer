#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:03:37 2025

@author: akihitomaruya
"""


import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

# --------------------------------------------------------
# Validation dataset + loader using SAME MaskGenerator
# --------------------------------------------------------

class SimMIMValDataset(torch.utils.data.Dataset):
    """
    Validation dataset that uses the SAME MaskGenerator
    as SimMIM training (mask_patch → model_patch upsample).
    """
    def __init__(self, root, config):
        self.config = config

        if config.MODEL.TYPE == "swin":
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == "vit":
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(int(config.DATA.IMG_SIZE * 256 / 224),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.DATA.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ])

        self.dataset = ImageFolder(root, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # EXACT SAME MASK LOGIC AS TRAINING
        mask = self.mask_generator()          # (Hm, Wm), int {0,1}
        mask = torch.from_numpy(mask).float() # float32 like training

        return img, mask, 0

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    is_dist, world, rank = _dist_info_safe()

    if is_dist:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        config.DATA.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dataloader

def build_val_loader_simmim(config, logger, val_path):
    if not val_path:
        return None

    logger.info(f"Build SimMIM val dataset from: {val_path}")

    dataset = SimMIMValDataset(val_path, config)

    is_dist, world, rank = _dist_info_safe()

    if is_dist:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader

# =========================================================
# CIFAR10 additions (DO NOT modify existing ImageFolder code)
# =========================================================
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

class EnsureRGB:
    """Pickle-safe replacement for T.Lambda(lambda img: img.convert('RGB') ...)"""
    def __call__(self, img):
        return img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img

# CIFAR10 stats (in [0,1] space)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.247, 0.243, 0.261) 


def _dist_info_safe():
    """Return (is_dist, world_size, rank) without assuming dist is initialized."""
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    return False, 1, 0


class SimMIMTransformCIFAR10:
    """
    CIFAR10 version of SimMIMTransform.
    Returns (img_tensor, mask_np) like the ImageFolder version.
    """
    def __init__(self, config):
        self.transform_img = T.Compose([
            EnsureRGB(), 
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(_CIFAR10_MEAN), std=torch.tensor(_CIFAR10_STD)),
        ])

        if config.MODEL.TYPE == "swin":
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == "vit":
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


class SimMIMValDatasetCIFAR10(torch.utils.data.Dataset):
    """
    CIFAR10 val dataset that matches SimMIMValDataset output:
      returns (img, mask, 0)
    Uses SAME MaskGenerator logic.
    """
    def __init__(self, root, config, *, download=False, indices=None):
        self.config = config

        if config.MODEL.TYPE == "swin":
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == "vit":
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

        self.transform = T.Compose([
            EnsureRGB(),  #  was a lambda (not picklable on macOS)
            T.Resize(int(config.DATA.IMG_SIZE * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.DATA.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(_CIFAR10_MEAN),
                std=torch.tensor(_CIFAR10_STD),
            ),
        ])

        # We use CIFAR10(train=True) and split into train/val by indices.
        base = CIFAR10(root=root, train=True, transform=self.transform, download=download)
        self.dataset = base if indices is None else Subset(base, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        mask = self.mask_generator()
        mask = torch.from_numpy(mask).float()
        return img, mask, 0


def _cifar10_split_indices(seed: int, train_frac: float):
    """
    Deterministic split of CIFAR10(train=True) indices into train/val.
    CIFAR10 train has 50,000 images.
    """
    n_total = 50000
    train_frac = float(train_frac)
    train_frac = min(max(train_frac, 0.05), 0.99)
    n_train = int(round(n_total * train_frac))

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx, val_idx


def build_loader_simmim_cifar10(config, logger, *, download=False, train_frac=0.8):
    """
    CIFAR10 train loader: yields (img, mask) just like build_loader_simmim.
    Uses CIFAR10(train=True) split.
    """
    is_dist, world, rank = _dist_info_safe()

    transform = SimMIMTransformCIFAR10(config)
    logger.info(f"Pre-train CIFAR10 data transform:\n{transform}")

    root = config.DATA.DATA_PATH
    base = CIFAR10(root=root, train=True, transform=transform, download=download)

    train_idx, _ = _cifar10_split_indices(seed=getattr(config, "SEED", 42), train_frac=train_frac)
    dataset = Subset(base, train_idx)

    logger.info(f"Build dataset (CIFAR10): train images = {len(dataset)}")

    if is_dist:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        config.DATA.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),  # (optional) avoids MPS warning
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dataloader


def build_val_loader_simmim_cifar10(config, logger, *, download=False, train_frac=0.8):
    """
    CIFAR10 val loader: yields (img, mask, 0) like build_val_loader_simmim.
    Uses CIFAR10(train=True) split (held-out portion).
    """
    is_dist, world, rank = _dist_info_safe()

    root = config.DATA.DATA_PATH
    _, val_idx = _cifar10_split_indices(seed=getattr(config, "SEED", 42), train_frac=train_frac)

    dataset = SimMIMValDatasetCIFAR10(root, config, download=download, indices=val_idx)

    logger.info(f"Build dataset (CIFAR10): val images = {len(dataset)}")

    if is_dist:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),  # (optional) avoids MPS warning
        drop_last=False,
    )
    return loader

