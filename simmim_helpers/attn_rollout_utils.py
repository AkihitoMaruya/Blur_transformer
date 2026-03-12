#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:33:57 2026

@author: akihitomaruya
"""

# simmim_helpers/attn_rollout_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

_LINEAR_MODES = {"linear", "bilinear", "bicubic", "trilinear"}


def seed_like_main(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
def attention_rollout(
    attn_maps: list[torch.Tensor],
    residual: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Deep -> shallow rollout for ViT Attention where:
      attn[q,k] = weight from query q to key k (softmax over k).

    attn_maps: list of [B,H,N,N] ordered shallow -> deep
    returns:   R [B,N,N] row-stochastic (rows sum to 1)

    Use CLS->spatial:  R[:,0,1:]
    """
    if not attn_maps:
        raise ValueError("attn_maps is empty")

    A0 = attn_maps[0]
    if A0.ndim != 4:
        raise ValueError(f"Expected [B,H,N,N], got {tuple(A0.shape)}")

    B, H, N, N2 = A0.shape
    if N != N2:
        raise ValueError(f"Expected square attention, got {tuple(A0.shape)}")

    dev = A0.device
    dtype = A0.dtype

    I = torch.eye(N, device=dev, dtype=dtype).unsqueeze(0)  # [1,N,N]
    R = torch.eye(N, device=dev, dtype=dtype).unsqueeze(0).expand(B, N, N).clone()

    # iterate deep -> shallow
    for A_h in reversed(attn_maps):
        A = A_h.mean(dim=1)  # [B,N,N]

        if residual != 0.0:
            A = A + residual * I

        # row-normalize (over keys)
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

        # compose deep->shallow under query->key convention
        R = A @ R

        # keep row-stochastic
        R = R / (R.sum(dim=-1, keepdim=True) + eps)

    return R




def reduce_tokens_to_spatial(vec_b_n: torch.Tensor, spatial_tokens: int) -> torch.Tensor:
    """
    Drop CLS if present; otherwise take the last spatial_tokens; pad if needed.
    vec_b_n: [B,N]
    return:  [B,spatial_tokens]
    """
    B, N = vec_b_n.shape
    if N == spatial_tokens + 1:
        return vec_b_n[:, 1:]
    if N > spatial_tokens:
        return vec_b_n[:, -spatial_tokens:]
    if N < spatial_tokens:
        pad = spatial_tokens - N
        return torch.cat([vec_b_n, vec_b_n.new_zeros(B, pad)], dim=1)
    return vec_b_n


def patchify_mean(x_b1hw: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Mean-pool [B,1,H,W] into patch grid [B,Hp*Wp].
    """
    B, _, H, W = x_b1hw.shape
    if (H % patch_size) != 0 or (W % patch_size) != 0:
        raise ValueError(f"H,W not divisible by patch_size: {(H,W)=} {patch_size=}")
    Hp, Wp = H // patch_size, W // patch_size
    x = x_b1hw.view(B, 1, Hp, patch_size, Wp, patch_size)
    x = x.mean(dim=(3, 5))  # [B,1,Hp,Wp]
    return x.view(B, Hp * Wp)


def tokens_to_map(weights_b_n: torch.Tensor, p_per_dim: int, patch_size: int) -> torch.Tensor:
    """
    [B,Nspatial] -> [B,1,H,W] via nearest upsample from patch grid.
    """
    B, N = weights_b_n.shape
    Ht = Wt = int(p_per_dim)
    if Ht * Wt != N:
        raise ValueError(f"N={N} != grid {Ht}x{Wt}")
    w = weights_b_n.view(B, 1, Ht, Wt)
    w = F.interpolate(w, scale_factor=int(patch_size), mode="nearest")
    return w


def percentile_norm(x_b1hw: torch.Tensor, lo: float = 1.0, hi: float = 99.0) -> torch.Tensor:
    """
    Per-image robust normalization into [0,1] using quantiles.
    """
    B = x_b1hw.size(0)
    xf = x_b1hw.view(B, -1)
    plo = torch.quantile(xf, lo / 100.0, dim=1, keepdim=True)
    phi = torch.quantile(xf, hi / 100.0, dim=1, keepdim=True)
    xs = (xf - plo) / (phi - plo + 1e-8)
    return xs.view_as(x_b1hw).clamp(0, 1)


def upsample_to(x_bchw: torch.Tensor, size_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """
    F.interpolate wrapper that only passes align_corners for linear modes.
    """
    if mode in _LINEAR_MODES:
        return F.interpolate(x_bchw, size=size_hw, mode=mode, align_corners=True)
    return F.interpolate(x_bchw, size=size_hw, mode=mode)


def pearson_r(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-example Pearson correlation.
    x,y: [B,K] -> r: [B]
    """
    x0 = x - x.mean(dim=1, keepdim=True)
    y0 = y - y.mean(dim=1, keepdim=True)
    num = (x0 * y0).sum(dim=1)
    den = torch.sqrt((x0.pow(2).sum(dim=1) + eps) * (y0.pow(2).sum(dim=1) + eps))
    return num / den


def unmask_bin_from_pixel_mask(
    pixel_mask_bool_b1hw: torch.Tensor,
    patch_size: int,
    fully_uncorrupted_thresh: float = 0.999,
) -> torch.Tensor:
    """
    pixel_mask_bool_b1hw: True=corrupted (same as your old corr.mask)
    returns:
      unmask_bin: [B,Nspatial] float, 1 only if FULL patch uncorrupted
    """
    unmask_pix = (~pixel_mask_bool_b1hw).float()                 # 1=uncorrupted
    unmask_patch = patchify_mean(unmask_pix, patch_size=patch_size)  # [B,N] in [0,1]
    unmask_bin = (unmask_patch > float(fully_uncorrupted_thresh)).float()
    return unmask_bin
