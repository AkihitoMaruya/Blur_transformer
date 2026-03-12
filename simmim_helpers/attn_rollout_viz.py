#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:34:31 2026

@author: akihitomaruya
"""
# simmim_helpers/attn_rollout_viz.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch


def colorize_heat_bw(
    heat_b1hw: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Map [B,1,H,W] in [0,1] to grayscale RGB where white = strong attention.

    Returns:
      [B,3,H,W] in [0,1] on (device,dtype).
    """
    if heat_b1hw.dim() != 4 or heat_b1hw.size(1) != 1:
        raise ValueError(f"heat_b1hw must be [B,1,H,W]. Got {tuple(heat_b1hw.shape)}")

    B, _, H, W = heat_b1hw.shape
    out = []

    for i in range(B):
        x = heat_b1hw[i, 0].detach().cpu().clamp(0, 1).numpy()  # [H,W]
        rgb = np.stack([x, x, x], axis=-1)                      # [H,W,3]
        out.append(torch.from_numpy(rgb).permute(2, 0, 1))      # [3,H,W]

    out_t = torch.stack(out, dim=0)  # [B,3,H,W]

    if device is None:
        device = heat_b1hw.device

    return out_t.to(device=device, dtype=dtype).clamp_(0, 1)


def draw_patch_borders_color(
    img_bchw: torch.Tensor,
    uncorr_mask_b1hw: torch.Tensor,
    patch_size: int,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.6,
    fully_uncorrupted_thresh: float = 0.999,
    border_px: int = 2,   # ← THICKER DEFAULT (was 1)
) -> torch.Tensor:
    """
    Draw color borders around FULLY UNCORRUPTED patches.

    Inputs:
      img_bchw:        [B,3,H,W] rollout image in [0,1]
      uncorr_mask_b1hw:[B,1,H,W] (bool or float) where 1/True = uncorrupted
      patch_size:      patch size in pixels
      border_px:       thickness of border in pixels (default=2)
    """
    if img_bchw.dim() != 4 or img_bchw.size(1) != 3:
        raise ValueError(f"img_bchw must be [B,3,H,W]. Got {tuple(img_bchw.shape)}")
    if uncorr_mask_b1hw.dim() != 4 or uncorr_mask_b1hw.size(1) != 1:
        raise ValueError(f"uncorr_mask_b1hw must be [B,1,H,W]. Got {tuple(uncorr_mask_b1hw.shape)}")

    B, _, H, W = img_bchw.shape
    if (H % patch_size) != 0 or (W % patch_size) != 0:
        raise ValueError(f"H,W not divisible by patch_size: {(H,W)=} {patch_size=}")

    Hp, Wp = H // patch_size, W // patch_size
    out = img_bchw.clone()
    col = out.new_tensor(color).view(3)

    uncorr_f = uncorr_mask_b1hw.float()
    bp = int(border_px)

    for b in range(B):
        for py in range(Hp):
            y0, y1 = py * patch_size, (py + 1) * patch_size
            for px in range(Wp):
                x0, x1 = px * patch_size, (px + 1) * patch_size

                patch = uncorr_f[b, 0, y0:y1, x0:x1]
                if not (patch.mean() > fully_uncorrupted_thresh):
                    continue

                # top
                out[b, :, y0:y0+bp, x0:x1] = (
                    (1 - alpha) * out[b, :, y0:y0+bp, x0:x1]
                    + alpha * col.view(3, 1, 1)
                )
                # bottom
                out[b, :, y1-bp:y1, x0:x1] = (
                    (1 - alpha) * out[b, :, y1-bp:y1, x0:x1]
                    + alpha * col.view(3, 1, 1)
                )
                # left
                out[b, :, y0:y1, x0:x0+bp] = (
                    (1 - alpha) * out[b, :, y0:y1, x0:x0+bp]
                    + alpha * col.view(3, 1, 1)
                )
                # right
                out[b, :, y0:y1, x1-bp:x1] = (
                    (1 - alpha) * out[b, :, y0:y1, x1-bp:x1]
                    + alpha * col.view(3, 1, 1)
                )

    return out.clamp_(0, 1)


