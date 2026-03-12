#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 15:21:32 2026

@author: akihitomaruya
"""


from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from simmim_helpers.msra_loader import build_msra_loader

from config import get_config
from models import build_model
from utils import load_checkpoint  # use repo loader
from simmim_helpers.knobs import set_model_knobs

from data import build_loader, build_val_loader  #  canonical training routing

# HF ImageNet32 loader (keep)
from simmim_helpers.imagenet32_hf_loader import build_imagenet32_loaders


# ============================================================
# NORMALIZATION CONSTANTS (EVAL CONTROL)
# ============================================================
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)



_IMAGENET_MEAN = tuple(float(x) for x in IMAGENET_DEFAULT_MEAN)
_IMAGENET_STD  = tuple(float(x) for x in IMAGENET_DEFAULT_STD)

_ALLOWED_UNEXPECTED_PREFIXES = ("blur._pyr.",)

NormName = Literal["cifar", "imagenet"]


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
def get_device() -> tuple[torch.device, bool]:
    """Priority: CUDA -> MPS -> CPU. Returns (device, use_cuda_bool)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


# ------------------------------------------------------------
# Minimal logger so knobs + loaders don't crash
# ------------------------------------------------------------
class SilentLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


# ------------------------------------------------------------
# Normalization selector (explicit)
# ------------------------------------------------------------
def _norm_from_name(norm: NormName) -> tuple[Tuple[float, float, float], Tuple[float, float, float], str]:
    n = str(norm).lower().strip()
    if n == "cifar":
        return _CIFAR10_MEAN, _CIFAR10_STD, "cifar"
    if n == "imagenet":
        return _IMAGENET_MEAN, _IMAGENET_STD, "imagenet"
    raise ValueError(f"norm must be 'cifar' or 'imagenet'. Got: {norm!r}")


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------
def _infer_cfg_path_from_ckpt(ckpt_path: Path) -> Path:
    """best.pt lives next to config.json (which actually contains YAML text)."""
    cfg = ckpt_path.parent / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"config.json not found next to checkpoint: {cfg}")
    return cfg


def _make_yacs_compatible_cfg_file(cfg_json_path: Path) -> Path:
    """Mirror config.json -> config__yacs.yaml so YACS will accept it."""
    txt = cfg_json_path.read_text()
    out_yaml = cfg_json_path.with_name(cfg_json_path.stem + "__yacs.yaml")
    if (not out_yaml.exists()) or (out_yaml.read_text() != txt):
        out_yaml.write_text(txt)
    return out_yaml


def _make_opts_from_overrides(overrides: Dict[str, Any]) -> list[str]:
    """
    IMPORTANT:
      - Do NOT pass bools through YACS merge_from_list (can type-mismatch in some setups).
      - Only pass int/float/str.
    """
    opts: list[str] = []
    for k, v in overrides.items():
        if isinstance(v, bool):
            continue
        opts.append(str(k))
        opts.append(str(v))
    return opts


# ------------------------------------------------------------
# Load diagnostics
# ------------------------------------------------------------
def _state_dict_load_report(
    msg,
    *,
    allow_unexpected_prefixes: Tuple[str, ...] = _ALLOWED_UNEXPECTED_PREFIXES,
) -> Dict[str, Any]:
    missing = list(getattr(msg, "missing_keys", [])) if msg is not None else []
    unexpected = list(getattr(msg, "unexpected_keys", [])) if msg is not None else []

    bad_unexpected = [k for k in unexpected if not k.startswith(allow_unexpected_prefixes)]
    safe_unexpected = [k for k in unexpected if k.startswith(allow_unexpected_prefixes)]

    return {
        "missing_keys_count": int(len(missing)),
        "unexpected_keys_count": int(len(unexpected)),
        "bad_unexpected_keys_count": int(len(bad_unexpected)),
        "safe_unexpected_keys_count": int(len(safe_unexpected)),
        "allow_unexpected_prefixes": list(allow_unexpected_prefixes),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "bad_unexpected_keys": bad_unexpected,
        "safe_unexpected_keys": safe_unexpected,
    }


def _blank_value_sanity(
    blank_value: float,
    *,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Dict[str, Any]:
    bv = float(blank_value)
    pixel_if_norm = [bv * std[c] + mean[c] for c in range(3)]
    norm_if_pixel = [(bv - mean[c]) / std[c] for c in range(3)]
    return {
        "raw_blank_value_from_config": bv,
        "assume_blank_is_pixel01_then_norm_equiv_per_channel": [float(v) for v in norm_if_pixel],
        "assume_blank_is_norm_then_pixel01_equiv_per_channel": [float(v) for v in pixel_if_norm],
        "mean_used": list(mean),
        "std_used": list(std),
    }


# ------------------------------------------------------------
# Force model mean/std buffers WITHOUT modifying simmim_blur.py
# ------------------------------------------------------------
def _force_model_norm_buffers(model: torch.nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> bool:
    if not (hasattr(model, "_mean") and hasattr(model, "_std")):
        return False
    with torch.no_grad():
        m = torch.tensor(mean, dtype=model._mean.dtype, device=model._mean.device).view(1, 3, 1, 1)
        s = torch.tensor(std,  dtype=model._std.dtype,  device=model._std.device).view(1, 3, 1, 1)
        model._mean.copy_(m)
        model._std.copy_(s)
    return True


# ------------------------------------------------------------
# EXACT main-like seeding (rank0)
# ------------------------------------------------------------
def _seed_like_main(config) -> None:
    seed0 = int(getattr(config, "SEED", 0))
    torch.manual_seed(seed0)
    np.random.seed(seed0)


# ------------------------------------------------------------
#  FIXED: Build loaders EXACTLY like main_simmim_blur.py for CIFAR10 & ImageNet
# ------------------------------------------------------------
def _build_loaders_exact_like_training(
    *,
    dataset: str,               # "cifar10" or "imagenet32" or "imagenet"
    config,
    batch_size: int,
    num_workers: int,
    data_path: Optional[Path] = None,          # NEW: matches args.data_path in training
    imagenet32_root: Optional[Path] = None,    # keep HF loader root
    # imagenet32-only
    val_ratio: float = 0.2,
    seed: int = 2025,
    norm: NormName = "imagenet",
) -> tuple[Optional[DataLoader], DataLoader]:
    """
    Returns (dl_train_or_None, dl_val).

    MATCHES main_simmim_blur.py for:
      - CIFAR10: build_loader(config,...,is_pretrain=True) + build_val_loader(..., val_path=None, is_pretrain=True)
      - ImageNet: build_loader(config,...,is_pretrain=True) + build_val_loader(..., val_path=<data_path>/val, is_pretrain=True)

    Keeps special path for:
      - ImageNet32: build_imagenet32_loaders(...)

    IMPORTANT: this uses data/__init__.py routing (your dataset_router),
    so it will log resolved roots and match your real training runs.
    """
    ds = str(dataset).lower().strip()

    config.defrost()
    config.DATA.DATASET = ds
    config.DATA.BATCH_SIZE = int(batch_size)
    config.DATA.NUM_WORKERS = int(num_workers)
    config.freeze()

    _seed_like_main(config)
    log = SilentLogger()

    # ----------------------------
    # CIFAR10 / ImageNet: use canonical training loaders
    # ----------------------------
    if ds in ("cifar10", "imagenet"):
        if data_path is None:
            raise ValueError(f"dataset='{ds}' requires data_path=... (same as args.data_path in training)")
        root = Path(data_path).expanduser().resolve()

        # EXACT training block: set DATA_PATH and CIFAR routing keys
        config.defrost()
        config.DATA.DATA_PATH = str(root)
        # these keys are read by data/__init__.py routing for CIFAR10
        if ds == "cifar10":
            if hasattr(config.DATA, "CIFAR_TRAIN_FRAC") and not hasattr(config.DATA, "_cifar_train_frac_set_by_eval"):
                # keep existing config value if present, but ensure it's float
                config.DATA.CIFAR_TRAIN_FRAC = float(getattr(config.DATA, "CIFAR_TRAIN_FRAC", 0.8))
            if hasattr(config.DATA, "CIFAR_DOWNLOAD") and not hasattr(config.DATA, "_cifar_download_set_by_eval"):
                config.DATA.CIFAR_DOWNLOAD = bool(getattr(config.DATA, "CIFAR_DOWNLOAD", False))
        config.freeze()

        dl_train = build_loader(config, log, is_pretrain=True)

        # EXACT training val_path logic
        val_path = None
        if ds != "cifar10":
            val_path = os.path.join(str(root), "val")
        dl_val = build_val_loader(config, log, val_path=val_path, is_pretrain=True)

        return dl_train, dl_val

    # ----------------------------
    # ImageNet32: keep existing HF loader path
    # ----------------------------
    if ds == "imagenet32":
        if imagenet32_root is None:
            raise ValueError("dataset='imagenet32' requires imagenet32_root=... (HF cache dir)")

        cache_dir = Path(imagenet32_root).expanduser().resolve()
        persist_dir = cache_dir / "_splits"

        mean, std, _ = _norm_from_name(norm)

        _dl_tr, dl_va, _, _ = build_imagenet32_loaders(
            cache_dir=cache_dir,
            persist_dir=persist_dir,
            config=config,
            return_masks=True,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=False,
            val_ratio=float(val_ratio),
            seed=int(seed),
            img_size=int(getattr(getattr(config, "DATA", None), "IMG_SIZE", 32)),
            normalize=True,
            mean=mean,
            std=std,
        )
        return _dl_tr, dl_va
    if ds == "msra":
        if data_path is None:
            raise ValueError("dataset='msra' requires data_path=... pointing to data_set/msra_cropped")
        root = Path(data_path).expanduser().resolve()
    
        mean, std, _ = _norm_from_name(norm)
    
        dl_val = build_msra_loader(
            root=root,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=False,
            img_size=int(getattr(getattr(config, "DATA", None), "IMG_SIZE", 32)),
            normalize=True,
            mean=mean,
            std=std,
        )
        return None, dl_val


    raise ValueError(f"Unsupported dataset '{dataset}'. Use 'cifar10', 'imagenet', or 'imagenet32'.")


# ------------------------------------------------------------
# Eval-only convenience loader (MODEL + TRAINING-STYLE LOADERS)
# ------------------------------------------------------------
def load_pretrained_model_for_eval(
    ckpt_path: str | Path,
    *,
    dataset: str,                  # "cifar10" or "imagenet" or "imagenet32"
    corr_type: str,                # "blur" or "blank"
    mask_ratio: float,
    depth: int,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    
    data_path: str | Path | None = None,       
    imagenet32_root: str | Path | None = None, # keep
    # eval normalization space
    norm: NormName = "cifar",
    # imagenet32-only
    val_ratio: float = 0.2,
    split_seed: int = 2025,
) -> Tuple[
    torch.nn.Module,
    Optional[DataLoader],
    DataLoader,
    Any,
    Dict[str, Any],
]:
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg_json = _infer_cfg_path_from_ckpt(ckpt_path)
    cfg_yaml = _make_yacs_compatible_cfg_file(cfg_json)

    ds = str(dataset).lower().strip()
    corr = str(corr_type).lower().strip()
    mean_used, std_used, norm_used = _norm_from_name(norm)

    class _Args:
        def __init__(self, cfg_file: str, opts: list[str]):
            self.cfg = cfg_file
            self.opts = opts
            self.local_rank = 0
            self.batch_size = None
            self.data_path = None
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.amp_opt_level = "O0"
            self.output = "output"
            self.tag = None
            self.lambda_span = 0.0
            self.pos_noise_std = 0.0

    overrides = {
        "DATA.DATASET": ds,
        "DATA.CORRUPTION": corr,
        "DATA.MASK_RATIO": float(mask_ratio),
        "DATA.BATCH_SIZE": int(batch_size),
        "DATA.NUM_WORKERS": int(num_workers),
    }
    args = _Args(str(cfg_yaml), _make_opts_from_overrides(overrides))
    config = get_config(args)

    # apply key overrides (match your training code style)
    config.defrost()
    config.DATA.DATASET = ds
    config.DATA.CORRUPTION = corr
    config.DATA.MASK_RATIO = float(mask_ratio)
    config.DATA.BATCH_SIZE = int(batch_size)
    config.DATA.NUM_WORKERS = int(num_workers)
    config.freeze()

    if device is None:
        device, _ = get_device()

    # ---------------- model ----------------
    model = build_model(config, is_pretrain=True).to(device)

    #  training-consistent checkpoint loader (PyTorch2.6 fix inside your function)
    config.defrost()
    config.MODEL.RESUME = str(ckpt_path)
    config.EVAL_MODE = True  # prevents optimizer/scheduler branch
    config.freeze()

    log = SilentLogger()
    load_checkpoint(config, model, optimizer=None, lr_scheduler=None, logger=log)

    # report (optional)
    # NOTE: load_checkpoint logs the msg; we can still compute a report if you want via strict=False load,
    # but we avoid re-loading here to keep behavior identical.

    # Truncate depth
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise RuntimeError("Model has no .encoder attribute")

    if hasattr(enc, "layers"):
        enc.layers = enc.layers[:depth]
    elif hasattr(enc, "blocks"):
        enc.blocks = enc.blocks[:depth]
    else:
        raise RuntimeError("Cannot truncate encoder depth: encoder has no layers/blocks")

    # Force model normalization buffers to match eval loader stats
    forced = _force_model_norm_buffers(model, mean_used, std_used)

    # knobs (safe defaults)
    set_model_knobs(model, pos_noise_std=0.0, lambda_span=0.0, logger=log)

    # ---------------- loaders ( FIXED) ----------------
    dl_tr, dl_va = _build_loaders_exact_like_training(
        dataset=ds,
        config=config,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        data_path=(Path(data_path) if data_path is not None else None),
        imagenet32_root=(Path(imagenet32_root) if imagenet32_root is not None else None),
        val_ratio=float(val_ratio),
        seed=int(split_seed),
        norm=norm,
    )

    blank_raw = float(getattr(config.DATA, "BLANK_VALUE", 0.0))
    meta = {
        "ckpt_path": str(ckpt_path),
        "dataset": ds,
        "corruption": corr,
        "mask_ratio": float(mask_ratio),
        "depth": int(depth),
        "norm": norm_used,
        "eval_norm_mean": list(mean_used),
        "eval_norm_std": list(std_used),
        "forced_model_mean_std_buffers": bool(forced),
        "blank_value": blank_raw,
        "blank_value_sanity": _blank_value_sanity(blank_raw, mean=mean_used, std=std_used),
        "img_size": int(getattr(getattr(config, "DATA", None), "IMG_SIZE", 32)),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
    }

    return model, dl_tr, dl_va, config, meta


# ============================================================
# Combined loader: build CIFAR10 + ImageNet32 from one ckpt load
# ============================================================

_COMBINED_PACK_CACHE: Dict[tuple, Dict[str, Any]] = {}

def load_pretrained_model_for_eval_cifar10_and_imagenet32(
    ckpt_path: str | Path,
    *,
    corr_type: str,                # "blur" or "blank"
    mask_ratio: float,
    depth: int,

    # shared loader params
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,

    # roots
    cifar_root: str | Path,
    imagenet32_root: str | Path,

    # eval normalization space (affects model buffers + im32 normalize)
    norm: NormName = "cifar",

    # imagenet32 split behavior
    val_ratio: float = 0.2,
    split_seed: int = 2025,

    # optional: return train loaders too (default False to save time)
    return_train: bool = False,
) -> Dict[str, Any]:
    """
    Load model+ckpt ONCE, then build BOTH:
      - CIFAR10 val loader
      - ImageNet32 val loader
    using the same config.

    Returns a dict:
      {
        "model": model,
        "config": config,
        "meta": meta,
        "cifar10": {"dl_tr": ..., "dl_va": ...},
        "imagenet32": {"dl_tr": ..., "dl_va": ...},
      }
    """

    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cifar_root = Path(cifar_root).expanduser().resolve()
    imagenet32_root = Path(imagenet32_root).expanduser().resolve()

    corr = str(corr_type).lower().strip()
    if corr not in ("blank", "blur"):
        raise ValueError(f"corr_type must be 'blank' or 'blur', got {corr_type!r}")

    # choose mean/std
    mean_used, std_used, norm_used = _norm_from_name(norm)

    # device
    if device is None:
        device, _ = get_device()

    # ------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------
    key = (
        str(ckpt_path),
        corr,
        float(mask_ratio),
        int(depth),
        int(batch_size),
        int(num_workers),
        str(device),
        str(norm_used),
        float(val_ratio),
        int(split_seed),
        str(cifar_root),
        str(imagenet32_root),
        bool(return_train),
    )
    if key in _COMBINED_PACK_CACHE:
        return _COMBINED_PACK_CACHE[key]

    # ------------------------------------------------------------
    # Build config ONCE (start from ckpt config.json)
    # ------------------------------------------------------------
    cfg_json = _infer_cfg_path_from_ckpt(ckpt_path)
    cfg_yaml = _make_yacs_compatible_cfg_file(cfg_json)

    class _Args:
        def __init__(self, cfg_file: str, opts: list[str]):
            self.cfg = cfg_file
            self.opts = opts
            self.local_rank = 0
            self.batch_size = None
            self.data_path = None
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.amp_opt_level = "O0"
            self.output = "output"
            self.tag = None
            self.lambda_span = 0.0
            self.pos_noise_std = 0.0

    # Note: we do NOT lock DATA.DATASET here because we need to build two loaders.
    overrides = {
        "DATA.CORRUPTION": corr,
        "DATA.MASK_RATIO": float(mask_ratio),
        "DATA.BATCH_SIZE": int(batch_size),
        "DATA.NUM_WORKERS": int(num_workers),
    }
    args = _Args(str(cfg_yaml), _make_opts_from_overrides(overrides))
    config = get_config(args)

    # apply shared overrides (keep DATASET mutable per-loader)
    config.defrost()
    config.DATA.CORRUPTION = corr
    config.DATA.MASK_RATIO = float(mask_ratio)
    config.DATA.BATCH_SIZE = int(batch_size)
    config.DATA.NUM_WORKERS = int(num_workers)
    config.freeze()

    # ------------------------------------------------------------
    # Build model ONCE + load ckpt ONCE
    # ------------------------------------------------------------
    model = build_model(config, is_pretrain=True).to(device)

    config.defrost()
    config.MODEL.RESUME = str(ckpt_path)
    config.EVAL_MODE = True
    config.freeze()

    log = SilentLogger()
    load_checkpoint(config, model, optimizer=None, lr_scheduler=None, logger=log)

    # truncate depth ONCE
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise RuntimeError("Model has no .encoder attribute")
    if hasattr(enc, "layers"):
        enc.layers = enc.layers[:depth]
    elif hasattr(enc, "blocks"):
        enc.blocks = enc.blocks[:depth]
    else:
        raise RuntimeError("Cannot truncate encoder depth: encoder has no layers/blocks")

    # force model mean/std buffers ONCE
    forced = _force_model_norm_buffers(model, mean_used, std_used)

    # knobs ONCE
    set_model_knobs(model, pos_noise_std=0.0, lambda_span=0.0, logger=log)

    # ------------------------------------------------------------
    # Build CIFAR10 loaders
    # ------------------------------------------------------------
    dl_tr_c10, dl_va_c10 = _build_loaders_exact_like_training(
        dataset="cifar10",
        config=config,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        data_path=cifar_root,
        imagenet32_root=None,
        val_ratio=float(val_ratio),     # unused for cifar10 path
        seed=int(split_seed),           # used by cifar split logic if any
        norm=norm,
    )
    if not return_train:
        dl_tr_c10 = None

    # ------------------------------------------------------------
    # Build ImageNet32 loaders
    # ------------------------------------------------------------
    dl_tr_im32, dl_va_im32 = _build_loaders_exact_like_training(
        dataset="imagenet32",
        config=config,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        data_path=None,
        imagenet32_root=imagenet32_root,
        val_ratio=float(val_ratio),
        seed=int(split_seed),
        norm=norm,
    )
    if not return_train:
        dl_tr_im32 = None

    # ------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------
    blank_raw = float(getattr(config.DATA, "BLANK_VALUE", 0.0))
    meta = {
        "ckpt_path": str(ckpt_path),
        "corruption": corr,
        "mask_ratio": float(mask_ratio),
        "depth": int(depth),
        "norm": norm_used,
        "eval_norm_mean": list(mean_used),
        "eval_norm_std": list(std_used),
        "forced_model_mean_std_buffers": bool(forced),
        "blank_value": blank_raw,
        "blank_value_sanity": _blank_value_sanity(blank_raw, mean=mean_used, std=std_used),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "cifar_root": str(cifar_root),
        "imagenet32_root": str(imagenet32_root),
        "return_train": bool(return_train),
    }

    pack = {
        "model": model,
        "config": config,
        "meta": meta,
        "cifar10": {"dl_tr": dl_tr_c10, "dl_va": dl_va_c10},
        "imagenet32": {"dl_tr": dl_tr_im32, "dl_va": dl_va_im32},
    }

    _COMBINED_PACK_CACHE[key] = pack
    return pack
