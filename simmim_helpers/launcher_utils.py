#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:44:23 2026

@author: akihitomaruya
"""
# simmim_helpers/launcher_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================
# Environment safety defaults (same as your launchers)
# =========================================================
def apply_env_safety_defaults() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")


# =========================================================
# Small helpers
# =========================================================
def on_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def pretty_ratio(r: float) -> str:
    return f"{int(round(float(r) * 100))}"


def fmt_sigma(s: float) -> str:
    s = float(s)
    if s.is_integer():
        return str(int(s))
    return str(s).replace(".", "p")


def print_cmd(args: List[str]) -> None:
    print(" ".join(shlex.quote(a) for a in args))


def run_cmd(args: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> int:
    if env is None:
        env = dict(os.environ)
    print("\n=== Launch ===")
    print_cmd(args)
    print("==============\n")
    return subprocess.run(args, cwd=str(cwd), env=env).returncode


def _yacs_bool_str(b: bool) -> str:
    return "True" if bool(b) else "False"


# =========================================================
# LEVELS: the critical fix
# =========================================================
def levels_to_cli(levels: List[Any]) -> str:
    """
    Return a python-literal list string that yacs can parse as a list.

    IMPORTANT:
      - ints must remain ints (no quotes)
      - "2" should become 2
      - strings like "residual_lowpass" remain quoted

    Examples:
      ["residual_lowpass", 2] -> "['residual_lowpass', 2]"
      ["residual_lowpass", "2"] -> "['residual_lowpass', 2]"
      ["residual_lowpass"] -> "['residual_lowpass']"
    """
    items: List[str] = []
    for x in levels:
        if isinstance(x, int) and not isinstance(x, bool):
            items.append(str(int(x)))
            continue

        s = str(x).strip()

        # numeric strings -> int literal
        if s.isdigit():
            items.append(str(int(s)))
            continue

        # normal string literal
        items.append(repr(s))

    return "[" + ", ".join(items) + "]"


def levels_to_tag(levels: List[Any]) -> str:
    """
    Filename/runname friendly tag:
      ["residual_lowpass", 2, 1] -> "residual_lowpass+2+1"
    """
    return "+".join(str(x).strip() for x in levels)


# =========================================================
# LR scaling + warmup
# =========================================================
def _infer_world_size() -> int:
    for k in ("WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(k, "").strip()
        if v.isdigit():
            return max(1, int(v))
    return 1


def levels_to_csv(levels: List[Any]) -> str:
    # ["residual_lowpass", 2] -> "residual_lowpass,2"
    out = []
    for x in levels:
        if isinstance(x, int) and not isinstance(x, bool):
            out.append(str(int(x)))
        else:
            s = str(x).strip()
            out.append(str(int(s)) if s.isdigit() else s)
    return ",".join(out)


def _get_cfg_batch_size(cfg_path: Path, default: int = 128) -> int:
    try:
        txt = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return int(default)

    m = re.search(r"\bBATCH_SIZE\s*:\s*([0-9]+)\b", txt)
    if m:
        return int(m.group(1))

    m = re.search(r"\bDATA\.BATCH_SIZE\s*:\s*([0-9]+)\b", txt)
    if m:
        return int(m.group(1))

    return int(default)


def _to_base512_lr(effective_lr: float, *, batch_size: int, world_size: int) -> float:
    """
    Convert an "effective LR" (tuned for global batch=512) into the LR
    appropriate for (batch_size_per_gpu * world_size).
    """
    denom = float(max(1, int(batch_size)) * max(1, int(world_size)))
    return float(effective_lr) * 512.0 / denom


def _suggest_lr_warmup(
    *,
    dataset: str,
    depth: Optional[int],
    epochs: int,
    head_only: bool = False,
) -> Dict[str, Any]:
    """
    CIFAR-style lighter recipe:
      - lighter than SimMIM/ImageNet defaults, but with a d>=6 branch that can
        realistically push CIFAR-10 >90% when combined with a decent backbone.
      - returns "effective" LRs (before base512 scaling).

    Notes:
      - For CIFAR d>=6 we use a more DeiT-ish augmentation/regularization:
        mixup+cutmix+label smoothing + higher WD + higher drop_path.
      - Keep your training pipeline consistent (batch scaling, scheduler, etc.).
    """
    ds = str(dataset).lower().strip()
    d = int(depth) if (depth is not None) else -1
    ep = int(epochs)

    # =========================================================
    # Global defaults (effective LR, i.e., before base512 scaling)
    # =========================================================
    base_lr = 5.0e-4
    warmup_lr = 5.0e-6
    min_lr = 1.0e-6
    warmup_epochs = 5

    layer_decay = 0.75
    weight_decay = 0.05
    drop_path = 0.10

    # Mixup/Cutmix knobs (strength)
    mixup = 0.0
    cutmix = 0.0

    # Mixup/Cutmix knobs (probabilities) + label smoothing
    mixup_prob = 0.0
    mixup_switch_prob = 0.0
    label_smoothing = 0.0

    # Optional augmentation knobs
    auto_augment = "none"
    color_jitter = 0.0
    reprob = 0.0
    remode = "pixel"
    recount = 1

    # =========================================================
    # CIFAR-10 / CIFAR-100
    # =========================================================
    if ds in ("cifar10", "cifar100"):
        # warmup schedule by total epochs
        if ep >= 200:
            warmup_epochs = 5
        elif ep >= 120:
            warmup_epochs = 4
        else:
            warmup_epochs = 3

        # CIFAR tends to like a higher warmup LR floor
        warmup_lr = 1.0e-4
        min_lr = 1.0e-5

        # -------------------------
        # Depth-specific recipe
        # -------------------------
        if d <= 2:
            # tiny models: minimal regularization/augmentation
            base_lr = 4.0e-3
            layer_decay = 1.00
            weight_decay = 0.02
            drop_path = 0.02

            mixup = 0.0
            cutmix = 0.0
            mixup_prob = 0.0
            mixup_switch_prob = 0.0
            label_smoothing = 0.0

            auto_augment = "none"
            color_jitter = 0.0
            reprob = 0.0

        elif d == 3:
            # light
            base_lr = 4.5e-3
            layer_decay = 0.90
            weight_decay = 0.02
            drop_path = 0.04

            mixup = 0.1
            cutmix = 0.0
            mixup_prob = 0.5
            mixup_switch_prob = 0.0
            label_smoothing = 0.05

            auto_augment = "rand-m2-mstd0.5-inc1"
            color_jitter = 0.2
            reprob = 0.05

        elif d == 4:
            # moderate but still "CIFAR-light"
            base_lr = 5.0e-3
            layer_decay = 0.80
            weight_decay = 0.02
            drop_path = 0.05

            mixup = 0.2
            cutmix = 0.0
            mixup_prob = 0.5
            mixup_switch_prob = 0.0
            label_smoothing = 0.05

            auto_augment = "rand-m3-mstd0.5-inc1"
            color_jitter = 0.3
            reprob = 0.10

        elif d == 5:
            # slightly stronger than d4, still much lighter than SimMIM default
            base_lr = 5.5e-3
            layer_decay = 0.75
            weight_decay = 0.02
            drop_path = 0.05

            mixup = 0.2
            cutmix = 0.0
            mixup_prob = 0.6
            mixup_switch_prob = 0.0
            label_smoothing = 0.05

            auto_augment = "rand-m4-mstd0.5-inc1"
            color_jitter = 0.4
            reprob = 0.10

        else:
            # d >= 6: slightly stronger, DeiT-ish CIFAR recipe
            # This is the branch to try when you're stuck in the high-80s.
            base_lr = 7.0e-3          # try 6e-3, 7e-3, 8e-3
            layer_decay = 0.75
            weight_decay = 0.05       # try 0.03–0.08
            drop_path = 0.10

            mixup = 0.4
            cutmix = 0.5              # try 0.5 then 1.0
            mixup_prob = 1.0
            mixup_switch_prob = 0.5
            label_smoothing = 0.10

            auto_augment = "rand-m6-mstd0.5-inc1"
            color_jitter = 0.4
            reprob = 0.10

        # -------------------------
        # Head-only override
        # -------------------------
        if head_only:
            base_lr = 0.03
            warmup_lr = 0.003
            min_lr = 1e-5
            warmup_epochs = 1
            weight_decay = 1e-4
            layer_decay = 1.0
            drop_path = 0.0

            mixup = 0.0
            cutmix = 0.0
            mixup_prob = 0.0
            mixup_switch_prob = 0.0
            label_smoothing = 0.0

            auto_augment = "none"
            color_jitter = 0.0
            reprob = 0.0

    # =========================================================
    # Everything else (ImageNet etc.) -> keep as-is for backward-compat
    # NOTE: build_finetune_cmd will NOT use these for ImageNet unless you ask it to.
    # =========================================================
    else:
        warmup_epochs = 5
        base_lr = 5.0e-4
        warmup_lr = min(5.0e-6, base_lr * 0.01)
        min_lr = 1e-6

        layer_decay = 0.65
        weight_decay = 0.05
        drop_path = 0.10

        mixup = 0.8
        cutmix = 1.0
        mixup_prob = 1.0
        mixup_switch_prob = 0.5
        label_smoothing = 0.1

        auto_augment = "rand-m9-mstd0.5-inc1"
        color_jitter = 0.4
        reprob = 0.25

        if head_only:
            base_lr = 0.03
            warmup_lr = 0.003
            min_lr = 1e-5
            warmup_epochs = 1
            weight_decay = 1e-4
            layer_decay = 1.0
            drop_path = 0.0

            mixup = 0.0
            cutmix = 0.0
            mixup_prob = 0.0
            mixup_switch_prob = 0.0
            label_smoothing = 0.0

            auto_augment = "none"
            color_jitter = 0.0
            reprob = 0.0

    return dict(
        warmup_epochs=int(warmup_epochs),
        base_lr=float(base_lr),
        warmup_lr=float(warmup_lr),
        min_lr=float(min_lr),
        layer_decay=float(layer_decay),
        weight_decay=float(weight_decay),
        drop_path=float(drop_path),

        mixup=float(mixup),
        cutmix=float(cutmix),
        mixup_prob=float(mixup_prob),
        mixup_switch_prob=float(mixup_switch_prob),
        label_smoothing=float(label_smoothing),

        # Optional: if you plumb these into YACS via --opts
        auto_augment=str(auto_augment),
        color_jitter=float(color_jitter),
        reprob=float(reprob),
        remode=str(remode),
        recount=int(recount),
    )


# =========================================================
# Config containers
# =========================================================
@dataclass(frozen=True)
class PretrainStatic:
    main: Path
    cfg: Path
    root: Path

    data_root: Path
    sweep_out: Path
    tag: str = "sweep"

    epochs: int = 200
    batch: int = 64
    workers: int = 8
    amp_level: str = "O0"

    patch_size: int = 4
    mask_patch_size: int = 4

    embed_dim: int = 384
    num_heads: int = 12
    mlp_ratio: int = 4

    lambda_span: float = 0.0
    pos_noise_std: float = 0.0
    val_freq: int = 1
    viz_freq: int = 1
    save_freq: int = 1

    dataset: str = "cifar10"
    cifar_train_frac: float = 0.8
    cifar_download: bool = True
    loss_on_full_image: bool = True

    force_num_workers: Optional[int] = None
    force_pin_memory: Optional[bool] = None

    # ---------------------------------------------------------
    # NEW (Option A): non-CIFAR (e.g., ImageNet) training overrides
    # These directly override YACS keys if provided (no base512 scaling).
    # CIFAR behavior is unchanged (these are ignored for CIFAR).
    # ---------------------------------------------------------
    override_train_epochs: Optional[int] = None
    override_warmup_epochs: Optional[int] = None
    override_base_lr: Optional[float] = None
    override_warmup_lr: Optional[float] = None
    override_min_lr: Optional[float] = None
    override_weight_decay: Optional[float] = None
    override_drop_path: Optional[float] = None
    override_layer_decay: Optional[float] = None  # usually for finetune; included for completeness


@dataclass(frozen=True)
class FinetuneStatic:
    main: Path
    main_headonly: Path
    cfg: Path
    root: Path

    data_root: Path
    finetune_out: Path
    tag: str = "finetune"

    finetune_epochs: int = 100
    save_every: int = 10
    amp_level: str = "O0"

    lambda_span: float = 0.0
    pos_noise_std: float = 0.0

    dataset: str = "imagenet"
    cifar_train_frac: float = 0.8
    cifar_download: bool = False
    always_force_seed: Optional[int] = None

    force_num_workers: Optional[int] = None
    force_pin_memory: Optional[bool] = None

    vit_patch_size: int = 4
    vit_embed_dim: int = 384
    vit_num_heads: int = 12
    vit_mlp_ratio: int = 4

    vit_use_mean_pooling: bool = False

    # ---------------------------------------------------------
    # (existing) launcher-enforced finetune aug/reg knobs
    # (kept for CIFAR behavior; for ImageNet we won't touch unless you override explicitly)
    # ---------------------------------------------------------
    aug_mixup_prob: Optional[float] = None
    aug_mixup_switch_prob: Optional[float] = None
    model_label_smoothing: Optional[float] = None

    finetune_corruption_override: Optional[str] = None

    # ---------------------------------------------------------
    # NEW (Option A): non-CIFAR (e.g., ImageNet) training overrides
    # These directly override YACS keys if provided (no base512 scaling).
    # CIFAR behavior is unchanged (these are ignored for CIFAR).
    # ---------------------------------------------------------
    override_train_epochs: Optional[int] = None
    override_warmup_epochs: Optional[int] = None
    override_base_lr: Optional[float] = None
    override_warmup_lr: Optional[float] = None
    override_min_lr: Optional[float] = None
    override_weight_decay: Optional[float] = None
    override_layer_decay: Optional[float] = None
    override_drop_path: Optional[float] = None

    # Optional ImageNet aug overrides (only used when provided AND not CIFAR)
    override_auto_augment: Optional[str] = None
    override_color_jitter: Optional[float] = None
    override_reprob: Optional[float] = None
    override_remode: Optional[str] = None
    override_recount: Optional[int] = None
    override_mixup: Optional[float] = None
    override_cutmix: Optional[float] = None
    override_mixup_prob: Optional[float] = None
    override_mixup_switch_prob: Optional[float] = None
    override_label_smoothing: Optional[float] = None


# =========================================================
# Run-name + corruption tags
# =========================================================
def build_corr_param_tag(corr_type: str, corr_params: Dict[str, Any]) -> str:
    ct = str(corr_type).lower().strip()

    if ct == "blank":
        v = float(corr_params.get("blank_value", 0.5))
        vtag = str(v).replace(".", "p")
        return f"blankv{vtag}"

    # UPDATED: FFT-Gaussian tag supports cutoff_ci (None => "nocut")
    if ct in ("gauss", "gaussian"):
        sigma_ci = float(
            corr_params.get("gaussian_sigma_ci", corr_params.get("gauss_sigma_ci", 2.0))
        )
        cutoff_ci = corr_params.get("gaussian_cutoff_ci", corr_params.get("gauss_cutoff_ci", None))
        s_tag = fmt_sigma(float(sigma_ci))
        if cutoff_ci is None:
            c_tag = "nocut"
        else:
            c_tag = f"c{fmt_sigma(float(cutoff_ci))}"
        return f"gaussci_s{s_tag}_{c_tag}"

    levels = corr_params.get("pyr_levels", ["residual_lowpass"])
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        levels = ["residual_lowpass"]
    lv_tag = levels_to_tag(list(levels))

    h = int(corr_params.get("pyr_height", 3))
    o = int(corr_params.get("pyr_order", 3))
    return f"pyr_{lv_tag}_h{h}_o{o}"


def make_run_name(
    *,
    corr_type: str,
    depth: int,
    mask_ratio: float,
    seed: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    corr_param_tag: str,
) -> str:
    ct = str(corr_type).lower().strip()
    if ct == "blank":
        prefix = "M"
    elif ct in ("gauss", "gaussian"):
        prefix = "G"
    else:
        prefix = "B"

    return (
        f"{prefix}-SimMIM_ViT"
        f"_d{int(depth)}"
        f"_ps{int(patch_size)}"
        f"_r{pretty_ratio(mask_ratio)}"
        f"_emb{int(embed_dim)}"
        f"_h{int(num_heads)}"
        f"_seed{int(seed)}"
        f"_{corr_param_tag}"
    )


# =========================================================
# Command builders
# =========================================================
def build_pretrain_cmd(
    *,
    S: PretrainStatic,
    seed: int,
    corr_type: str,
    corr_params: Dict[str, Any],
    depth: int,
    mask_ratio: float,
    out_dir: Path,
) -> List[str]:
    python = os.environ.get("PYTHON", "") or "python"
    ct = str(corr_type).lower().strip()

    ds = str(S.dataset).lower().strip()
    is_cifar = (ds in ("cifar10", "cifar100"))

    args: List[str] = [
        python,
        "-u",
        str(S.main),
        "--cfg",
        str(S.cfg),
        "--data-path",
        str(S.data_root),
        "--dataset",
        str(S.dataset),
    ]

    # CIFAR-only CLI flags (keep exact behavior)
    if is_cifar:
        args += ["--cifar-train-frac", str(float(S.cifar_train_frac))]
        if bool(S.cifar_download):
            args += ["--cifar-download"]

    args += [
        "--output",
        str(out_dir),
        "--tag",
        str(S.tag),
        "--amp-opt-level",
        str(S.amp_level),
        "--val-freq",
        str(int(S.val_freq)),
        "--viz-freq",
        str(int(S.viz_freq)),
    ]

    if bool(S.loss_on_full_image):
        args += ["--loss-on-full-image"]

    args += ["--lambda-span", str(float(S.lambda_span))]
    args += ["--pos-noise-std", str(float(S.pos_noise_std))]

    # corr kind (note: gauss uses corruption=blur; details set via --opts)
    cli_corr = "blur" if ct in ("gauss", "gaussian") else ct
    args += ["--corruption", cli_corr]

    # CIFAR: launcher sweeps mask_ratio (keep behavior).
    # non-CIFAR: by default, do NOT force --mask-ratio (let YAML decide)
    if is_cifar:
        args += ["--mask-ratio", str(float(mask_ratio))]

    args += ["--mask-patch-size", str(int(S.mask_patch_size))]

    if cli_corr == "blank":
        args += ["--blank-value", str(float(corr_params.get("blank_value", 0.5)))]
    else:
        if ct not in ("gauss", "gaussian"):
            levels = corr_params.get("pyr_levels", ["residual_lowpass"])
            if not isinstance(levels, (list, tuple)) or len(levels) == 0:
                levels = ["residual_lowpass"]
            args += ["--blur-levels", levels_to_csv(list(levels))]
            args += ["--blur-height", str(int(corr_params.get("pyr_height", 3)))]
            args += ["--blur-order", str(int(corr_params.get("pyr_order", 3)))]

    effective_workers = (
        int(S.force_num_workers)
        if (S.force_num_workers is not None)
        else (0 if on_macos() else int(S.workers))
    )
    effective_pin = (
        bool(S.force_pin_memory)
        if (S.force_pin_memory is not None)
        else (False if on_macos() else True)
    )

    opts: List[str] = [
        "SEED", str(int(seed)),
        "SAVE_FREQ", str(int(S.save_freq)),
        "DATA.DATASET", str(S.dataset),
        "DATA.BATCH_SIZE", str(int(S.batch)),
        "DATA.NUM_WORKERS", str(int(effective_workers)),
        "DATA.PIN_MEMORY", _yacs_bool_str(bool(effective_pin)),
        "TRAIN.EPOCHS", str(int(S.epochs)),
        "TRAIN.ACCUMULATION_STEPS", "1",
        "MODEL.TYPE", "vit",
        "MODEL.VIT.PATCH_SIZE", str(int(S.patch_size)),
        "MODEL.VIT.EMBED_DIM", str(int(S.embed_dim)),
        "MODEL.VIT.DEPTH", str(int(depth)),
        "MODEL.VIT.NUM_HEADS", str(int(S.num_heads)),
        "MODEL.VIT.MLP_RATIO", str(int(S.mlp_ratio)),
    ]

    # CIFAR-only hard overrides (keep exact behavior)
    if is_cifar:
        opts += [
            "DATA.IMG_SIZE", "32",
            "MODEL.NUM_CLASSES", "10",
            "TRAIN.WARMUP_EPOCHS", "5",
        ]

    # non-CIFAR: apply explicit overrides ONLY if provided (Option A)
    if not is_cifar:
        if S.override_train_epochs is not None:
            opts += ["TRAIN.EPOCHS", str(int(S.override_train_epochs))]
        if S.override_warmup_epochs is not None:
            opts += ["TRAIN.WARMUP_EPOCHS", str(int(S.override_warmup_epochs))]
        if S.override_base_lr is not None:
            opts += ["TRAIN.BASE_LR", str(float(S.override_base_lr))]
        if S.override_warmup_lr is not None:
            opts += ["TRAIN.WARMUP_LR", str(float(S.override_warmup_lr))]
        if S.override_min_lr is not None:
            opts += ["TRAIN.MIN_LR", str(float(S.override_min_lr))]
        if S.override_weight_decay is not None:
            opts += ["TRAIN.WEIGHT_DECAY", str(float(S.override_weight_decay))]
        if S.override_drop_path is not None:
            opts += ["MODEL.DROP_PATH_RATE", str(float(S.override_drop_path))]
        if S.override_layer_decay is not None:
            opts += ["TRAIN.LAYER_DECAY", str(float(S.override_layer_decay))]

    # FFT-Gaussian backend (cycles/image) + optional cutoff
    if ct in ("gauss", "gaussian"):
        sigma_ci = float(corr_params.get("gaussian_sigma_ci", corr_params.get("gauss_sigma_ci", 2.0)))
        cutoff_ci = corr_params.get("gaussian_cutoff_ci", corr_params.get("gauss_cutoff_ci", None))

        opts += [
            "DATA.BLUR_TYPE", "gaussian",
            "DATA.BLUR_GAUSS_SIGMA_CI", str(float(sigma_ci)),
        ]
        if cutoff_ci is not None:
            opts += ["DATA.BLUR_GAUSS_CUTOFF_CI", str(float(cutoff_ci))]

    args += ["--opts"] + opts
    return args


def build_finetune_cmd(
    *,
    S: FinetuneStatic,
    seed: int,
    corr_type: str,
    corr_params: Dict[str, Any],
    depth: int,
    pretrained_ckpt: Path,
    out_dir: Path,
    head_only: bool,
) -> List[str]:
    python = os.environ.get("PYTHON", "") or "python"
    main_script = S.main_headonly if bool(head_only) else S.main

    ct = str(corr_type).lower().strip()
    forced_seed = int(S.always_force_seed) if (S.always_force_seed is not None) else int(seed)

    ds = str(S.dataset).lower().strip()
    is_cifar = (ds == "cifar10")

    # Prior behavior: gauss piggybacks on corruption="blur"
    ft_corr = "blur" if ct in ("gauss", "gaussian") else ct
    if getattr(S, "finetune_corruption_override", None) is not None:
        ft_corr = str(getattr(S, "finetune_corruption_override")).lower().strip()

    # CIFAR path keeps exact behavior: use recipe + base512 scaling
    if is_cifar:
        lrw = _suggest_lr_warmup(
            dataset=str(S.dataset),
            depth=int(depth),
            epochs=int(S.finetune_epochs),
            head_only=bool(head_only),
        )

        ft_world = _infer_world_size()
        ft_bs = _get_cfg_batch_size(Path(S.cfg), default=128)

        base_lr_512 = _to_base512_lr(float(lrw["base_lr"]), batch_size=ft_bs, world_size=ft_world)
        warmup_lr_512 = _to_base512_lr(float(lrw["warmup_lr"]), batch_size=ft_bs, world_size=ft_world)
        min_lr_512 = _to_base512_lr(float(lrw["min_lr"]), batch_size=ft_bs, world_size=ft_world)

        effective_workers = (
            int(S.force_num_workers) if (S.force_num_workers is not None) else (0 if on_macos() else 8)
        )
        effective_pin = (
            bool(S.force_pin_memory) if (S.force_pin_memory is not None) else (False if on_macos() else True)
        )

        _ov_mixup_prob = getattr(S, "aug_mixup_prob", None)
        _ov_mixup_switch_prob = getattr(S, "aug_mixup_switch_prob", None)
        _ov_label_smoothing = getattr(S, "model_label_smoothing", None)

        mixup_prob = float(_ov_mixup_prob) if (_ov_mixup_prob is not None) else float(lrw.get("mixup_prob", 0.0))
        mixup_switch_prob = (
            float(_ov_mixup_switch_prob)
            if (_ov_mixup_switch_prob is not None)
            else float(lrw.get("mixup_switch_prob", 0.0))
        )
        label_smoothing = (
            float(_ov_label_smoothing)
            if (_ov_label_smoothing is not None)
            else float(lrw.get("label_smoothing", 0.0))
        )

        _ov_auto_augment = getattr(S, "aug_auto_augment", None)
        _ov_color_jitter = getattr(S, "aug_color_jitter", None)
        _ov_reprob = getattr(S, "aug_reprob", None)
        _ov_remode = getattr(S, "aug_remode", None)
        _ov_recount = getattr(S, "aug_recount", None)

        auto_augment = str(_ov_auto_augment) if (_ov_auto_augment is not None) else str(lrw.get("auto_augment", "none"))
        color_jitter = float(_ov_color_jitter) if (_ov_color_jitter is not None) else float(lrw.get("color_jitter", 0.0))
        reprob = float(_ov_reprob) if (_ov_reprob is not None) else float(lrw.get("reprob", 0.0))
        remode = str(_ov_remode) if (_ov_remode is not None) else str(lrw.get("remode", "pixel"))
        recount = int(_ov_recount) if (_ov_recount is not None) else int(lrw.get("recount", 1))

        args: List[str] = [
            python,
            "-u",
            str(main_script),
            "--cfg",
            str(S.cfg),
            "--data-path",
            str(S.data_root),
            "--output",
            str(out_dir),
            "--tag",
            str(S.tag),
            "--amp-opt-level",
            str(S.amp_level),
            "--local_rank",
            "0",
            "--lambda-span",
            str(float(0.0 if head_only else S.lambda_span)),
            "--pos-noise-std",
            str(float(S.pos_noise_std)),
            "--pretrained",
            str(pretrained_ckpt),
            "--opts",
            "SEED", str(int(forced_seed)),
            "TRAIN.EPOCHS", str(int(S.finetune_epochs)),
            "SAVE_FREQ", str(int(S.save_every)),
            "DATA.DATASET", str(S.dataset),
            "DATA.CORRUPTION", str(ft_corr),
            "DATA.MASK_RATIO", "0.0",
            "DATA.NUM_WORKERS", str(int(effective_workers)),
            "DATA.PIN_MEMORY", _yacs_bool_str(bool(effective_pin)),
            "MODEL.TYPE", "vit",
            "MODEL.NUM_CLASSES", "10",
            "MODEL.VIT.PATCH_SIZE", str(int(S.vit_patch_size)),
            "MODEL.VIT.EMBED_DIM", str(int(S.vit_embed_dim)),
            "MODEL.VIT.DEPTH", str(int(depth)),
            "MODEL.VIT.NUM_HEADS", str(int(S.vit_num_heads)),
            "MODEL.VIT.MLP_RATIO", str(int(round(float(S.vit_mlp_ratio)))),
            "MODEL.VIT.USE_MEAN_POOLING", _yacs_bool_str(bool(S.vit_use_mean_pooling)),

            "DATA.CIFAR_FINETUNE_USE_SPLIT", "True",
            "DATA.CIFAR_TRAIN_FRAC", str(float(S.cifar_train_frac)),
            "DATA.CIFAR_DOWNLOAD", _yacs_bool_str(bool(S.cifar_download)),
            "DATA.IMG_SIZE", "32",

            "TRAIN.WARMUP_EPOCHS", str(int(lrw["warmup_epochs"])),
            "TRAIN.BASE_LR", str(float(base_lr_512)),
            "TRAIN.WARMUP_LR", str(float(warmup_lr_512)),
            "TRAIN.MIN_LR", str(float(min_lr_512)),
            "TRAIN.LAYER_DECAY", str(float(lrw.get("layer_decay", 1.0))),
            "TRAIN.WEIGHT_DECAY", str(float(lrw.get("weight_decay", 0.05))),
            "MODEL.DROP_PATH_RATE", str(float(lrw.get("drop_path", 0.1))),

            "AUG.MIXUP", str(float(lrw.get("mixup", 0.0))),
            "AUG.CUTMIX", str(float(lrw.get("cutmix", 0.0))),
            "AUG.MIXUP_PROB", str(float(mixup_prob)),
            "AUG.MIXUP_SWITCH_PROB", str(float(mixup_switch_prob)),
            "MODEL.LABEL_SMOOTHING", str(float(label_smoothing)),

            "AUG.AUTO_AUGMENT", str(auto_augment),
            "AUG.COLOR_JITTER", str(float(color_jitter)),
            "AUG.REPROB", str(float(reprob)),
            "AUG.REMODE", str(remode),
            "AUG.RECOUNT", str(int(recount)),

            "TRAIN.ACCUMULATION_STEPS", "1",
        ]

    # ImageNet (and all non-CIFAR): keep YAML behavior by default,
    # apply explicit Option-A overrides ONLY if provided (no base512 scaling).
    else:
        effective_workers = (
            int(S.force_num_workers) if (S.force_num_workers is not None) else (0 if on_macos() else 8)
        )
        effective_pin = (
            bool(S.force_pin_memory) if (S.force_pin_memory is not None) else (False if on_macos() else True)
        )

        args = [
            python,
            "-u",
            str(main_script),
            "--cfg",
            str(S.cfg),
            "--data-path",
            str(S.data_root),
            "--output",
            str(out_dir),
            "--tag",
            str(S.tag),
            "--amp-opt-level",
            str(S.amp_level),
            "--local_rank",
            "0",
            "--lambda-span",
            str(float(0.0 if head_only else S.lambda_span)),
            "--pos-noise-std",
            str(float(S.pos_noise_std)),
            "--pretrained",
            str(pretrained_ckpt),
            "--opts",
            "SEED", str(int(forced_seed)),
            "SAVE_FREQ", str(int(S.save_every)),
            "DATA.DATASET", str(S.dataset),
            "DATA.CORRUPTION", str(ft_corr),
            "DATA.MASK_RATIO", "0.0",
            "DATA.NUM_WORKERS", str(int(effective_workers)),
            "DATA.PIN_MEMORY", _yacs_bool_str(bool(effective_pin)),
            "MODEL.TYPE", "vit",
            "MODEL.VIT.PATCH_SIZE", str(int(S.vit_patch_size)),
            "MODEL.VIT.EMBED_DIM", str(int(S.vit_embed_dim)),
            "MODEL.VIT.DEPTH", str(int(depth)),
            "MODEL.VIT.NUM_HEADS", str(int(S.vit_num_heads)),
            "MODEL.VIT.MLP_RATIO", str(int(round(float(S.vit_mlp_ratio)))),
            "MODEL.VIT.USE_MEAN_POOLING", _yacs_bool_str(bool(S.vit_use_mean_pooling)),
            "TRAIN.ACCUMULATION_STEPS", "1",
        ]

        # Option A overrides (non-CIFAR only)
        if S.override_train_epochs is not None:
            args += ["TRAIN.EPOCHS", str(int(S.override_train_epochs))]
        if S.override_warmup_epochs is not None:
            args += ["TRAIN.WARMUP_EPOCHS", str(int(S.override_warmup_epochs))]
        if S.override_base_lr is not None:
            args += ["TRAIN.BASE_LR", str(float(S.override_base_lr))]
        if S.override_warmup_lr is not None:
            args += ["TRAIN.WARMUP_LR", str(float(S.override_warmup_lr))]
        if S.override_min_lr is not None:
            args += ["TRAIN.MIN_LR", str(float(S.override_min_lr))]
        if S.override_weight_decay is not None:
            args += ["TRAIN.WEIGHT_DECAY", str(float(S.override_weight_decay))]
        if S.override_layer_decay is not None:
            args += ["TRAIN.LAYER_DECAY", str(float(S.override_layer_decay))]
        if S.override_drop_path is not None:
            args += ["MODEL.DROP_PATH_RATE", str(float(S.override_drop_path))]

        # Optional ImageNet aug overrides (only when provided)
        if S.override_auto_augment is not None:
            args += ["AUG.AUTO_AUGMENT", str(S.override_auto_augment)]
        if S.override_color_jitter is not None:
            args += ["AUG.COLOR_JITTER", str(float(S.override_color_jitter))]
        if S.override_reprob is not None:
            args += ["AUG.REPROB", str(float(S.override_reprob))]
        if S.override_remode is not None:
            args += ["AUG.REMODE", str(S.override_remode)]
        if S.override_recount is not None:
            args += ["AUG.RECOUNT", str(int(S.override_recount))]

        if S.override_mixup is not None:
            args += ["AUG.MIXUP", str(float(S.override_mixup))]
        if S.override_cutmix is not None:
            args += ["AUG.CUTMIX", str(float(S.override_cutmix))]
        if S.override_mixup_prob is not None:
            args += ["AUG.MIXUP_PROB", str(float(S.override_mixup_prob))]
        if S.override_mixup_switch_prob is not None:
            args += ["AUG.MIXUP_SWITCH_PROB", str(float(S.override_mixup_switch_prob))]
        if S.override_label_smoothing is not None:
            args += ["MODEL.LABEL_SMOOTHING", str(float(S.override_label_smoothing))]

    # corruption params (both branches)
    if ft_corr == "blank":
        args += ["DATA.BLANK_VALUE", str(float(corr_params.get("blank_value", 0.5)))]

    elif ct in ("gauss", "gaussian"):
        sigma_ci = float(corr_params.get("gaussian_sigma_ci", corr_params.get("gauss_sigma_ci", 2.0)))
        cutoff_ci = corr_params.get("gaussian_cutoff_ci", corr_params.get("gauss_cutoff_ci", None))

        args += ["DATA.BLUR_TYPE", "gaussian", "DATA.BLUR_GAUSS_SIGMA_CI", str(float(sigma_ci))]
        if cutoff_ci is not None:
            args += ["DATA.BLUR_GAUSS_CUTOFF_CI", str(float(cutoff_ci))]

    else:
        args += ["DATA.BLUR_HEIGHT", str(int(corr_params.get("pyr_height", 3)))]
        if "pyr_levels" in corr_params:
            lv = corr_params["pyr_levels"]
            if isinstance(lv, (list, tuple)) and len(lv) > 0:
                args += ["DATA.BLUR_LEVELS", levels_to_csv(list(lv))]
        if "pyr_order" in corr_params:
            args += ["DATA.BLUR_ORDER", str(int(corr_params["pyr_order"]))]

    return args


