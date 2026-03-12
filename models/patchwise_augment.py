#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:58:30 2025

@author: akihitomaruya
"""
# models/patchwise_augment.py
# --------------------------------------------------------
# Patchwise pyramid blur / blank driven by an explicit patch mask.
# Written by Akihito Maruya
#
# UPDATE (FFT Gaussian, no kernel, DC-centered):
#   - blur_type="gaussian" means *frequency-domain* Gaussian low-pass
#   - sigma is in frequency domain (cycles/image)
#   - cutoff is cycles/image threshold:
#       if cutoff_ci = 6, then r>6 cycles/image => set to 0
#   - uses fftshift/ifftshift to apply a *DC-centered* radial filter correctly
# --------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import ast

import torch
import torch.nn as nn

from plenoptic.simulate import SteerablePyramidFreq


# -------------------------
# Helpers
# -------------------------

def _assert_bchw(x: torch.Tensor) -> Tuple[int, int, int, int]:
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        raise ValueError(
            f"Expected x as (B,C,H,W) torch.Tensor, got {type(x)} {tuple(getattr(x,'shape',()))}"
        )
    return x.shape  # (B,C,H,W)


def _normalize_mask_small(mask_small: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Returns mask_small as (B,1,Hb,Wb) bool on `device`.
    Accepts (B,Hb,Wb), (B,1,Hb,Wb).
    """
    if mask_small.dim() == 3:
        mask_small = mask_small.unsqueeze(1)  # (B,1,Hb,Wb)
    elif mask_small.dim() == 4 and mask_small.shape[1] == 1:
        pass
    else:
        raise ValueError(
            f"mask_small must be (B,Hb,Wb) or (B,1,Hb,Wb), got {tuple(mask_small.shape)}"
        )

    mask_small = mask_small.to(device=device, non_blocking=True)
    if mask_small.dtype != torch.bool:
        mask_small = mask_small > 0.5
    return mask_small


def _expand_mask(mask_small_b1hw: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    mask_small_b1hw: (B,1,Hb,Wb) bool -> (B,1,H,W) bool by repeating patches
    """
    mask = mask_small_b1hw.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask[..., :H, :W]


# -------------------------
# Configs
# -------------------------

@dataclass
class PyrBlurConfig:
    patch_size: int = 32

    blur_type: str = "pyramid"  # "pyramid" or "gaussian"

    # pyramid params
    levels: Sequence[Union[str, int]] = ("residual_lowpass",)
    height: int = 5
    order: int = 3

    # FFT Gaussian params (only used if blur_type == "gaussian")
    gaussian_sigma_ci: float = 4.0
    gaussian_cutoff_ci: Optional[float] = None  # <-- None => no cutoff

    # optional pyramid-space noise
    use_noise: bool = False
    noise_std: float = 0.1
    noise_seed: Optional[int] = None


@dataclass
class BlankConfig:
    patch_size: int = 32
    value: float = 0.5  # grayscale value in [0,1]


# -------------------------
# Modules
# -------------------------

class PatchwisePyrBlurFromMask(nn.Module):
    """
    Apply blur only on patches specified by mask_small.

    Backends:
      - blur_type="pyramid": SteerablePyramidFreq recon with cfg.levels
      - blur_type="gaussian": FFT-domain Gaussian low-pass w/ hard cutoff (DC-centered via fftshift)

    Inputs:
      x01:        (B,C,H,W) (typically in [0,1], but we do NOT clamp)
      mask_small: (B,Hb,Wb) or (B,1,Hb,Wb) where Hb=H/patch_size, Wb=W/patch_size
                 True => blur this patch, False => keep original
    Output:
      x_out: (B,C,H,W)
    """
    def __init__(self, cfg: PyrBlurConfig):
        super().__init__()
        self.cfg = cfg

        # pyramid cache
        self._pyr: Optional[SteerablePyramidFreq] = None
       
        # FFT-gaussian filter cache (DC-centered)
        self._gfilt: Optional[torch.Tensor] = None
        self._gfilt_sig: Optional[Tuple[int, int, float, Optional[float], str, torch.dtype]] = None


        self._printed_levels_once = False

    # -------------------------
    # Pyramid backend
    # -------------------------

    def _ensure_pyr(self, H: int, W: int, device: torch.device):
        if self._pyr is None:
            self._pyr = SteerablePyramidFreq(
                height=int(self.cfg.height),
                image_shape=[H, W],
                order=int(self.cfg.order),
            ).to(device)

    @staticmethod
    def _parse_levels(levels_in):
        """
        Accept:
          - "residual_lowpass,2"
          - ["residual_lowpass", 2]
          - [["residual_lowpass", 2]]  (unwrap)
          - "['residual_lowpass', 2]"
        """
        lv = levels_in

        if isinstance(lv, (list, tuple)) and len(lv) == 1 and isinstance(lv[0], (list, tuple)):
            lv = lv[0]

        if isinstance(lv, (list, tuple)) and len(lv) == 1 and isinstance(lv[0], str):
            s0 = lv[0].strip()
            if (s0.startswith("[") and s0.endswith("]")) or (s0.startswith("(") and s0.endswith(")")):
                try:
                    lv = ast.literal_eval(s0)
                except Exception:
                    lv = s0
            else:
                lv = s0

        if isinstance(lv, str):
            s = lv.strip()
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                out = []
                for p in parts:
                    out.append(int(p) if p.isdigit() else p)
                return out
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    lv = ast.literal_eval(s)
                except Exception:
                    return [s]
            else:
                return [s]

        if not isinstance(lv, (list, tuple)):
            lv = [lv]

        out = []
        for t in lv:
            if isinstance(t, int) and not isinstance(t, bool):
                out.append(int(t))
            elif isinstance(t, str):
                ts = t.strip()
                out.append(int(ts) if ts.isdigit() else ts)
            else:
                out.append(t)
        return out

    @torch.no_grad()
    def _lowpass_pyramid(self, x01: torch.Tensor) -> torch.Tensor:
        B, C, H, W = _assert_bchw(x01)
        self._ensure_pyr(H, W, x01.device)
        assert self._pyr is not None

        x32 = x01.to(dtype=torch.float32)
        coeffs = self._pyr.forward(x32)

        levels = self._parse_levels(self.cfg.levels)
        if not self._printed_levels_once:
            print("[PYR] cfg.levels raw:", self.cfg.levels, type(self.cfg.levels))
            print("[PYR] levels parsed :", levels, [type(x) for x in levels])
            self._printed_levels_once = True

        out = self._pyr.recon_pyr(coeffs, levels=levels)
        return out.to(dtype=x01.dtype)

    # -------------------------
    # FFT Gaussian backend (cycles/image, DC-centered via fftshift)
    # -------------------------

    def _ensure_gfilt(self, H: int, W: int, device: torch.device, dtype: torch.dtype):
        """
        Build (1,1,H,W) real-valued *DC-centered* radial Gaussian low-pass.
        Optionally apply a hard cutoff if cfg.gaussian_cutoff_ci is not None.
    
        Units: cycles/image.
          - Nyquist radius = min(H,W)/2  (e.g., 32 -> 16)
        """
        sigma_ci = float(self.cfg.gaussian_sigma_ci)
        cutoff_ci = self.cfg.gaussian_cutoff_ci  # may be None
    
        if sigma_ci <= 0:
            raise ValueError(f"gaussian_sigma_ci must be > 0, got {sigma_ci}")
    
        nyq_ci = 0.5 * float(min(H, W))
        if cutoff_ci is not None:
            cutoff_ci = float(cutoff_ci)
            if not (0.0 < cutoff_ci <= nyq_ci):
                raise ValueError(f"gaussian_cutoff_ci must be in (0, {nyq_ci}] or None, got {cutoff_ci}")
    
        sig = (H, W, sigma_ci, cutoff_ci, device.type, dtype)
        if self._gfilt is not None and self._gfilt_sig == sig:
            return
    
        # fftfreq is cycles/pixel in [-0.5,0.5). Multiply by size -> cycles/image.
        fy_ci = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype) * float(H)  # (H,)
        fx_ci = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype) * float(W)  # (W,)
    
        yy, xx = torch.meshgrid(fy_ci, fx_ci, indexing="ij")  # (H,W)
        yy = torch.fft.fftshift(yy, dim=(-2, -1))
        xx = torch.fft.fftshift(xx, dim=(-2, -1))
    
        r_ci = torch.sqrt(xx * xx + yy * yy)
    
        g = torch.exp(-(r_ci * r_ci) / (2.0 * sigma_ci * sigma_ci))
    
        # Only apply cutoff when requested
        if cutoff_ci is not None:
            g = g * (r_ci < cutoff_ci)
    
        self._gfilt = g.view(1, 1, H, W)
        self._gfilt_sig = sig


    @torch.no_grad()
    def _blur_gaussian_fft(self, x01: torch.Tensor) -> torch.Tensor:
        """
        FFT-domain Gaussian low-pass with hard cutoff (cycles/image).
        Uses fftshift/ifftshift so the filter is DC-centered.
        NO clamping.
        """
        B, C, H, W = _assert_bchw(x01)

        x32 = x01.to(dtype=torch.float32)

        self._ensure_gfilt(H, W, x32.device, x32.dtype)
        assert self._gfilt is not None  # (1,1,H,W), DC-centered

        # FFT
        X = torch.fft.fft2(x32, dim=(-2, -1))

        # shift so DC is at center
        Xs = torch.fft.fftshift(X, dim=(-2, -1))

        # apply filter
        Ys = Xs * self._gfilt

        # unshift
        Y = torch.fft.ifftshift(Ys, dim=(-2, -1))

        # iFFT
        y = torch.fft.ifft2(Y, dim=(-2, -1)).real

        return y.to(dtype=x01.dtype)

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, x01: torch.Tensor, mask_small: torch.Tensor) -> torch.Tensor:
        B, C, H, W = _assert_bchw(x01)
        ps = int(self.cfg.patch_size)
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be divisible by cfg.patch_size={ps}. Got H={H}, W={W}")

        mask_small = _normalize_mask_small(mask_small, x01.device)  # (B,1,Hb,Wb)
        Hb, Wb = H // ps, W // ps
        if mask_small.shape[-2:] != (Hb, Wb):
            raise ValueError(
                f"mask_small spatial must be (Hb,Wb)=({Hb},{Wb}) for patch_size={ps}, got {tuple(mask_small.shape[-2:])}"
            )

        bt = str(getattr(self.cfg, "blur_type", "pyramid")).lower().strip()
        if bt not in ("pyramid", "gaussian"):
            raise ValueError(f"cfg.blur_type must be 'pyramid' or 'gaussian', got {self.cfg.blur_type!r}")

        if bt == "gaussian":
            low = self._blur_gaussian_fft(x01)
        else:
            low = self._lowpass_pyramid(x01)

        mask_pix = _expand_mask(mask_small, ps, H, W)  # (B,1,H,W)
        return torch.where(mask_pix, low, x01)


class PatchwiseBlankFromMask(nn.Module):
    """
    Replace patches specified by mask_small with a constant value.

    Inputs:
      x01:        (B,C,H,W)
      mask_small: (B,Hb,Wb) or (B,1,Hb,Wb)
                 True => blank this patch, False => keep original
    Output:
      x_out: (B,C,H,W)
    """
    def __init__(self, cfg: BlankConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x01: torch.Tensor, mask_small: torch.Tensor) -> torch.Tensor:
        B, C, H, W = _assert_bchw(x01)
        ps = int(self.cfg.patch_size)
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be divisible by cfg.patch_size={ps}. Got H={H}, W={W}")

        mask_small = _normalize_mask_small(mask_small, x01.device)  # (B,1,Hb,Wb)
        Hb, Wb = H // ps, W // ps
        if mask_small.shape[-2:] != (Hb, Wb):
            raise ValueError(
                f"mask_small spatial must be (Hb,Wb)=({Hb},{Wb}) for patch_size={ps}, got {tuple(mask_small.shape[-2:])}"
            )

        val = float(self.cfg.value)
        mask_pix = _expand_mask(mask_small, ps, H, W)  # (B,1,H,W)
        fill = torch.full_like(x01, val)
        return torch.where(mask_pix, fill, x01)


# -------------------------
# Builders from YACS config
# -------------------------
def build_pyr_blur_from_config(config) -> PatchwisePyrBlurFromMask:
    noise_seed = int(getattr(config.DATA, "BLUR_NOISE_SEED", -1))
    noise_seed = None if noise_seed < 0 else noise_seed

    # --- IMPORTANT: allow YAML null -> Python None for cutoff ---
    cutoff_ci = getattr(config.DATA, "BLUR_GAUSS_CUTOFF_CI", None)
    cutoff_ci = None if cutoff_ci is None else float(cutoff_ci)

    cfg = PyrBlurConfig(
        patch_size=int(getattr(config.DATA, "BLUR_PATCH_SIZE", config.DATA.MASK_PATCH_SIZE)),

        blur_type=str(getattr(config.DATA, "BLUR_TYPE", "pyramid")),

        # Pyramid params
        levels=getattr(config.DATA, "BLUR_LEVELS", ("residual_lowpass",)),
        height=int(getattr(config.DATA, "BLUR_HEIGHT", 5)),
        order=int(getattr(config.DATA, "BLUR_ORDER", 3)),

        # FFT-gaussian params (cycles/image)
        gaussian_sigma_ci=float(getattr(config.DATA, "BLUR_GAUSS_SIGMA_CI", 7.0)),
        gaussian_cutoff_ci=cutoff_ci,

        # Pyramid-space noise (only applied in pyramid mode)
        use_noise=bool(getattr(config.DATA, "BLUR_USE_NOISE", False)),
        noise_std=float(getattr(config.DATA, "BLUR_NOISE_STD", 0.1)),
        noise_seed=noise_seed,
    )
    return PatchwisePyrBlurFromMask(cfg)



def build_blank_from_config(config) -> PatchwiseBlankFromMask:
    blank_patch = int(getattr(config.DATA, "BLUR_PATCH_SIZE", config.DATA.MASK_PATCH_SIZE))
    blank_val = float(getattr(config.DATA, "BLANK_VALUE", 0.5))

    cfg = BlankConfig(
        patch_size=blank_patch,
        value=blank_val,
    )
    return PatchwiseBlankFromMask(cfg)

