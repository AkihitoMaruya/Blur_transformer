"""
combined_launcher.py
------------------------------------------------------------
Importable combined launcher:
  seeds -> corr_types -> corr_params -> depths -> mask_ratios -> PRETRAIN -> FINETUNE

Adds checkpoint selection for finetune:
  - pretrain_ckpt_mode="best" (default)
  - pretrain_ckpt_mode="last"
  - pretrain_ckpt_mode="epoch" with:
        pretrain_epoch=100
    or  pretrain_epochs=[50,100,200]  (runs multiple finetunes per pretrain run)

Designed to be run from Spyder by importing:
    from combined_launcher import run_launcher

KEY FIX (your issue):
- pyr_level_sets supports mixed types: e.g. [["residual_lowpass", 2], ["residual_lowpass"]]
- We coerce numeric strings to ints, but DO NOT stringify ints upstream.

GAUSSIAN UPDATE:
- Supports FFT-Gaussian params:
    gaussian_sigma_ci (float, cycles/image)  [REQUIRED]
    gaussian_cutoff_ci (Optional[float], cycles/image)  [None => no cutoff]
- Accepts corr_type "gaussian" / "gauss" (and also "blur" with blur_type chosen inside training).
- `gauss_conditions` should be a list of dicts like:
    [
      {"gaussian_sigma_ci": 2.0, "gaussian_cutoff_ci": 6.0},
      {"gaussian_sigma_ci": 2.0, "gaussian_cutoff_ci": None},
    ]

OPTION A (ImageNet behavior; CIFAR unchanged):
- CIFAR: keep existing behavior (launcher-driven CIFAR overrides).
- ImageNet / non-CIFAR: YAML is the default source of truth.
  Only if you explicitly pass override_* values into run_launcher (and thus into PretrainStatic/FinetuneStatic)
  will the launcher override YAML via --opts (implemented inside launcher_utils.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simmim_helpers.launcher_utils import (
    apply_env_safety_defaults,
    on_macos,
    PretrainStatic,
    FinetuneStatic,
    build_corr_param_tag,
    make_run_name,
    build_pretrain_cmd,
    build_finetune_cmd,
    run_cmd,
)


# ------------------------------------------------------------
# Local checkpoint finder (supports best/last/epoch)
# Uses your pretrain folder convention:
#   <run_dir>/simmim_pretrain/<tag>/{best.pth, ckpt_epoch_*.pth, ...}
# ------------------------------------------------------------
def find_checkpoint(
    run_dir: Path,
    *,
    tag: str,
    mode: str = "best",              # "best" | "last" | "epoch"
    epoch: Optional[int] = None,     # required if mode=="epoch"
) -> Optional[Path]:
    ckpt_dir = run_dir / "simmim_pretrain" / tag
    search_root = ckpt_dir if ckpt_dir.is_dir() else run_dir

    exts = (".pth", ".pt", ".ckpt")

    def _exists(stem: str) -> Optional[Path]:
        for ext in exts:
            p = search_root / f"{stem}{ext}"
            if p.is_file():
                return p
        return None

    def _newest_epoch_ckpt() -> Optional[Path]:
        ckpts = list(search_root.rglob("ckpt_epoch_*.pth")) + list(search_root.rglob("ckpt_epoch_*.pt"))
        if not ckpts:
            ckpts = list(run_dir.rglob("ckpt_epoch_*.pth")) + list(run_dir.rglob("ckpt_epoch_*.pt"))
        if not ckpts:
            return None

        def epnum(p: Path) -> int:
            import re
            m = re.search(r"ckpt_epoch_(\d+)", p.name)
            return int(m.group(1)) if m else -1

        ckpts.sort(key=epnum)
        return ckpts[-1]

    m = str(mode).lower().strip()

    if m == "epoch":
        if epoch is None:
            raise ValueError("find_checkpoint(mode='epoch') requires epoch != None")
        cand = _exists(f"ckpt_epoch_{int(epoch)}")
        if cand is not None:
            return cand

        import re
        patt = re.compile(rf"(ckpt[_-]epoch[_-]{int(epoch)})|(epoch[_-]{int(epoch)})", re.IGNORECASE)
        hits: List[Path] = []
        for ext in ("*.pth", "*.pt", "*.ckpt"):
            hits += [p for p in run_dir.rglob(ext) if patt.search(p.name)]
        if hits:
            hits.sort(key=lambda p: (len(p.parts), p.name))
            return hits[-1]
        return None

    if m == "best":
        cand = _exists("best")
        if cand is not None:
            return cand
        return _newest_epoch_ckpt()

    if m == "last":
        cand = _newest_epoch_ckpt()
        if cand is not None:
            return cand
        return _exists("best")

    cand = _exists("best")
    if cand is not None:
        return cand
    return _newest_epoch_ckpt()


# ------------------------------------------------------------
# Corr param set builder (supports blur list-of-lists)
# ------------------------------------------------------------
def _coerce_level_token(x: Any) -> Any:
    """
    Keep ints as ints; turn numeric strings like "2" into int(2);
    keep normal strings as strings.
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
        return s
    return x


def _normalize_gauss_condition(gc_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize/validate FFT-Gaussian condition dict.

    Expected keys (preferred):
      - gaussian_sigma_ci: float (required)
      - gaussian_cutoff_ci: Optional[float] (None => no cutoff)

    We also accept legacy keys:
      - gauss_sigma_ci -> gaussian_sigma_ci
      - gauss_cutoff_ci -> gaussian_cutoff_ci

    Any extra keys are preserved and passed through.
    """
    gc = dict(gc_in)

    # allow legacy aliases
    if "gaussian_sigma_ci" not in gc and "gauss_sigma_ci" in gc:
        gc["gaussian_sigma_ci"] = gc.pop("gauss_sigma_ci")
    if "gaussian_cutoff_ci" not in gc and "gauss_cutoff_ci" in gc:
        gc["gaussian_cutoff_ci"] = gc.pop("gauss_cutoff_ci")

    if "gaussian_sigma_ci" not in gc:
        raise ValueError(f"Gaussian condition missing required key 'gaussian_sigma_ci': {gc_in}")

    gc["gaussian_sigma_ci"] = float(gc["gaussian_sigma_ci"])

    if "gaussian_cutoff_ci" not in gc:
        gc["gaussian_cutoff_ci"] = None
    else:
        if gc["gaussian_cutoff_ci"] is None:
            gc["gaussian_cutoff_ci"] = None
        else:
            gc["gaussian_cutoff_ci"] = float(gc["gaussian_cutoff_ci"])

    return gc


def _build_corr_param_sets(
    corr_type: str,
    *,
    pyr_level_sets: Optional[Sequence[Sequence[Any]]] = None,
    pyr_heights: Optional[Sequence[int]] = None,
    pyr_orders: Optional[Sequence[int]] = None,
    blank_values: Optional[Sequence[float]] = None,
    gauss_conditions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    ct = str(corr_type).lower().strip()

    if ct == "blur":
        if not pyr_level_sets:
            pyr_level_sets = [["residual_lowpass"]]
        if not pyr_heights:
            pyr_heights = [3]
        if not pyr_orders:
            pyr_orders = [3]

        out: List[Dict[str, Any]] = []
        for lv in pyr_level_sets:
            lv_clean = [_coerce_level_token(t) for t in list(lv)]
            for h in pyr_heights:
                for o in pyr_orders:
                    out.append(dict(pyr_levels=lv_clean, pyr_height=int(h), pyr_order=int(o)))
        return out

    if ct == "blank":
        if not blank_values:
            blank_values = [0.5]
        return [dict(blank_value=float(v)) for v in blank_values]

    if ct in ("gauss", "gaussian"):
        if not gauss_conditions:
            gauss_conditions = [dict(gaussian_sigma_ci=2.0, gaussian_cutoff_ci=None)]
        return [_normalize_gauss_condition(gc) for gc in gauss_conditions]

    raise ValueError(f"Unknown corr_type: {corr_type}")


# ------------------------------------------------------------
# Main callable
# ------------------------------------------------------------
def run_launcher(
    *,
    # required
    root: str | Path,

    # pretrain
    pre_cfg: str | Path,
    pre_main: str | Path,
    pre_data_root: str | Path,
    sweep_out: str | Path,

    # finetune
    ft_cfg: str | Path,
    ft_main: str | Path,
    ft_main_headonly: str | Path,
    ft_data_root: str | Path,
    finetune_out: str | Path,

    # sweep lists
    seeds: Sequence[int],
    corr_types: Sequence[str],
    depths: Sequence[int],
    mask_ratios: Sequence[float],

    # blur conditions (list-of-lists; supports ints)
    pyr_level_sets: Optional[Sequence[Sequence[Any]]] = None,
    pyr_heights: Optional[Sequence[int]] = None,
    pyr_orders: Optional[Sequence[int]] = None,

    # blank conditions
    blank_values: Optional[Sequence[float]] = None,

    # gaussian conditions (FFT-domain; sigma/cutoff in cycles/image)
    gauss_conditions: Optional[Sequence[Dict[str, Any]]] = None,

    # behavior flags
    head_only: bool = False,

    # pooling choice for finetune features
    #   "cls"  -> CLS token (USE_MEAN_POOLING=False)
    #   "mean" -> mean pooling (USE_MEAN_POOLING=True)
    ft_pooling: str = "cls",

    # checkpoint selection for finetune
    pretrain_ckpt_mode: str = "best",                 # "best" | "last" | "epoch"
    pretrain_epoch: Optional[int] = None,             # used if pretrain_ckpt_mode="epoch"
    pretrain_epochs: Optional[Sequence[int]] = None,  # if provided, overrides pretrain_epoch

    # -------------------------
    # pretrain hyperparams (CIFAR behavior remains launcher-driven)
    # -------------------------
    pre_tag: str = "sweep",
    pre_epochs: int = 200,
    pre_batch: int = 64,
    pre_workers: int = 0,        # SAFE default for macOS/Spyder
    pre_amp: str = "O0",         # SAFE default for MPS

    pre_patch_size: int = 4,
    # NEW: mask patch size decoupled from patch_size (ImageNet default is 32)
    # Keep default behavior by defaulting to None => uses pre_patch_size.
    pre_mask_patch_size: Optional[int] = None,

    pre_embed_dim: int = 384,
    pre_num_heads: int = 12,
    pre_mlp_ratio: int = 4,
    pre_lambda_span: float = 0.0,
    pre_pos_noise_std: float = 0.0,
    pre_val_freq: int = 1,
    pre_viz_freq: int = 1,
    pre_save_freq: int = 1,
    pre_dataset: str = "cifar10",
    pre_cifar_train_frac: float = 0.8,
    pre_cifar_download: bool = True,
    pre_loss_on_full_image: bool = True,

    # -------------------------
    # Option A: ImageNet/non-CIFAR pretrain overrides (YAML is default; None => no override)
    # These are consumed inside launcher_utils.build_pretrain_cmd().
    # -------------------------
    pre_override_train_epochs: Optional[int] = None,
    pre_override_warmup_epochs: Optional[int] = None,
    pre_override_base_lr: Optional[float] = None,
    pre_override_warmup_lr: Optional[float] = None,
    pre_override_min_lr: Optional[float] = None,
    pre_override_weight_decay: Optional[float] = None,
    pre_override_drop_path: Optional[float] = None,

    # -------------------------
    # finetune hyperparams
    # -------------------------
    ft_tag: str = "finetune",
    ft_epochs: int = 100,
    ft_save_every: int = 10,
    ft_amp: str = "O0",          # SAFE default for MPS
    ft_lambda_span: float = 0.0,
    ft_pos_noise_std: float = 0.0,
    ft_dataset: str = "imagenet",
    ft_cifar_train_frac: float = 0.8,
    ft_cifar_download: bool = False,
    ft_always_force_seed: Optional[int] = None,

    # CIFAR finetune override knobs (keep None unless you want to override your defaults)
    ft_mixup_prob: Optional[float] = None,
    ft_mixup_switch_prob: Optional[float] = None,
    ft_label_smoothing: Optional[float] = None,

    ft_corruption_override: Optional[str] = None,

    # -------------------------
    # Option A: ImageNet/non-CIFAR finetune overrides (YAML is default; None => no override)
    # These are consumed inside launcher_utils.build_finetune_cmd().
    # -------------------------
    ft_override_train_epochs: Optional[int] = None,
    ft_override_warmup_epochs: Optional[int] = None,
    ft_override_base_lr: Optional[float] = None,
    ft_override_warmup_lr: Optional[float] = None,
    ft_override_min_lr: Optional[float] = None,
    ft_override_weight_decay: Optional[float] = None,
    ft_override_layer_decay: Optional[float] = None,
    ft_override_drop_path: Optional[float] = None,

    # Optional augmentation overrides for non-CIFAR (or if you want to override YAML explicitly)
    ft_override_auto_augment: Optional[str] = None,
    ft_override_color_jitter: Optional[float] = None,
    ft_override_reprob: Optional[float] = None,
    ft_override_remode: Optional[str] = None,
    ft_override_recount: Optional[int] = None,

    pre_train: bool = True,
) -> None:
    apply_env_safety_defaults()

    # Force MPS-safe defaults when on macOS (Spyder/import-friendly)
    if on_macos():
        pre_workers = 0
        pre_amp = "O0"
        ft_amp = "O0"

    ROOT = Path(root).expanduser().resolve()

    # pooling parsing
    pooling = str(ft_pooling).lower().strip()
    if pooling not in ("cls", "mean"):
        raise ValueError(f"ft_pooling must be 'cls' or 'mean', got: {ft_pooling}")
    vit_use_mean_pooling = (pooling == "mean")

    # default: preserve old behavior (mask_patch_size == patch_size) unless explicitly provided
    if pre_mask_patch_size is None:
        pre_mask_patch_size = int(pre_patch_size)

    PRE = PretrainStatic(
        main=Path(pre_main).expanduser().resolve(),
        cfg=Path(pre_cfg).expanduser().resolve(),
        root=ROOT,
        data_root=Path(pre_data_root).expanduser().resolve(),
        sweep_out=Path(sweep_out).expanduser().resolve(),
        tag=str(pre_tag),

        epochs=int(pre_epochs),
        batch=int(pre_batch),
        workers=int(pre_workers),
        amp_level=str(pre_amp),

        patch_size=int(pre_patch_size),
        mask_patch_size=int(pre_mask_patch_size),

        embed_dim=int(pre_embed_dim),
        num_heads=int(pre_num_heads),
        mlp_ratio=int(pre_mlp_ratio),

        lambda_span=float(pre_lambda_span),
        pos_noise_std=float(pre_pos_noise_std),

        val_freq=int(pre_val_freq),
        viz_freq=int(pre_viz_freq),
        save_freq=int(pre_save_freq),

        dataset=str(pre_dataset),
        cifar_train_frac=float(pre_cifar_train_frac),
        cifar_download=bool(pre_cifar_download),
        loss_on_full_image=bool(pre_loss_on_full_image),

        # MPS-safe dataloader behavior
        force_num_workers=(0 if on_macos() else None),
        force_pin_memory=(False if on_macos() else None),

        # Option A: ImageNet/non-CIFAR overrides (launcher_utils decides when to apply)
        override_train_epochs=(int(pre_override_train_epochs) if pre_override_train_epochs is not None else None),
        override_warmup_epochs=(int(pre_override_warmup_epochs) if pre_override_warmup_epochs is not None else None),
        override_base_lr=(float(pre_override_base_lr) if pre_override_base_lr is not None else None),
        override_warmup_lr=(float(pre_override_warmup_lr) if pre_override_warmup_lr is not None else None),
        override_min_lr=(float(pre_override_min_lr) if pre_override_min_lr is not None else None),
        override_weight_decay=(float(pre_override_weight_decay) if pre_override_weight_decay is not None else None),
        override_drop_path=(float(pre_override_drop_path) if pre_override_drop_path is not None else None),
    )

    # IMPORTANT: copy PRE geometry into FT so finetune model matches ckpt
    FT = FinetuneStatic(
        main=Path(ft_main).expanduser().resolve(),
        main_headonly=Path(ft_main_headonly).expanduser().resolve(),
        cfg=Path(ft_cfg).expanduser().resolve(),
        root=ROOT,
        data_root=Path(ft_data_root).expanduser().resolve(),
        finetune_out=Path(finetune_out).expanduser().resolve(),
        tag=str(ft_tag),

        finetune_epochs=int(ft_epochs),
        save_every=int(ft_save_every),
        amp_level=str(ft_amp),

        lambda_span=float(ft_lambda_span),
        pos_noise_std=float(ft_pos_noise_std),

        dataset=str(ft_dataset),
        cifar_train_frac=float(ft_cifar_train_frac),
        cifar_download=bool(ft_cifar_download),
        always_force_seed=(int(ft_always_force_seed) if ft_always_force_seed is not None else None),

        # MPS-safe dataloader behavior
        force_num_workers=(0 if on_macos() else None),
        force_pin_memory=(False if on_macos() else None),

        # geometry overrides (ckpt-compatible)
        vit_patch_size=int(PRE.patch_size),
        vit_embed_dim=int(PRE.embed_dim),
        vit_num_heads=int(PRE.num_heads),
        vit_mlp_ratio=int(PRE.mlp_ratio),

        # pooling choice
        vit_use_mean_pooling=bool(vit_use_mean_pooling),

        # CIFAR-specific explicit override knobs (keep None unless user provided a value)
        aug_mixup_prob=(None if ft_mixup_prob is None else float(ft_mixup_prob)),
        aug_mixup_switch_prob=(None if ft_mixup_switch_prob is None else float(ft_mixup_switch_prob)),
        model_label_smoothing=(None if ft_label_smoothing is None else float(ft_label_smoothing)),

        finetune_corruption_override=(str(ft_corruption_override) if ft_corruption_override is not None else None),

        # Option A: ImageNet/non-CIFAR overrides (launcher_utils decides when to apply)
        override_train_epochs=(int(ft_override_train_epochs) if ft_override_train_epochs is not None else None),
        override_warmup_epochs=(int(ft_override_warmup_epochs) if ft_override_warmup_epochs is not None else None),
        override_base_lr=(float(ft_override_base_lr) if ft_override_base_lr is not None else None),
        override_warmup_lr=(float(ft_override_warmup_lr) if ft_override_warmup_lr is not None else None),
        override_min_lr=(float(ft_override_min_lr) if ft_override_min_lr is not None else None),
        override_weight_decay=(float(ft_override_weight_decay) if ft_override_weight_decay is not None else None),
        override_layer_decay=(float(ft_override_layer_decay) if ft_override_layer_decay is not None else None),
        override_drop_path=(float(ft_override_drop_path) if ft_override_drop_path is not None else None),

        override_auto_augment=(str(ft_override_auto_augment) if ft_override_auto_augment is not None else None),
        override_color_jitter=(float(ft_override_color_jitter) if ft_override_color_jitter is not None else None),
        override_reprob=(float(ft_override_reprob) if ft_override_reprob is not None else None),
        override_remode=(str(ft_override_remode) if ft_override_remode is not None else None),
        override_recount=(int(ft_override_recount) if ft_override_recount is not None else None),
    )

    PRE.sweep_out.mkdir(parents=True, exist_ok=True)
    FT.finetune_out.mkdir(parents=True, exist_ok=True)

    # build checkpoint specs
    ckpt_specs: List[Tuple[str, Optional[int]]] = []
    if pretrain_epochs is not None:
        ckpt_specs = [("epoch", int(e)) for e in pretrain_epochs]
    elif pretrain_epoch is not None:
        ckpt_specs = [("epoch", int(pretrain_epoch))]
    else:
        ckpt_specs = [(str(pretrain_ckpt_mode).lower().strip(), None)]

    failures: List[str] = []

    # seeds -> corr_types -> corr_params -> depths -> mask_ratios -> pretrain -> finetune
    for seed in seeds:
        seed = int(seed)

        for corr_type in corr_types:
            corr_type = str(corr_type).lower().strip()

            corr_param_sets = _build_corr_param_sets(
                corr_type,
                pyr_level_sets=pyr_level_sets,
                pyr_heights=pyr_heights,
                pyr_orders=pyr_orders,
                blank_values=blank_values,
                gauss_conditions=gauss_conditions,
            )

            # HARD sanity check: never allow "2" (string) in levels
            if corr_type == "blur":
                for cp in corr_param_sets:
                    lv = cp.get("pyr_levels", None)
                    if not isinstance(lv, list):
                        raise ValueError(f"pyr_levels must be a list, got {type(lv)}: {lv}")
                    if len(lv) >= 2 and lv[1] == "2":
                        raise ValueError(f"BUG: second level is string '2' (should be int): {lv}")

            # gaussian_cutoff_ci is either None or float
            if corr_type in ("gauss", "gaussian"):
                for cp in corr_param_sets:
                    if "gaussian_sigma_ci" not in cp:
                        raise ValueError(f"Gaussian corr_params missing gaussian_sigma_ci: {cp}")
                    if cp.get("gaussian_cutoff_ci", None) is not None and not isinstance(cp["gaussian_cutoff_ci"], float):
                        raise ValueError(
                            f"Gaussian gaussian_cutoff_ci must be float or None, got: {cp['gaussian_cutoff_ci']}"
                        )

            for corr_params in corr_param_sets:
                corr_tag = build_corr_param_tag(corr_type, corr_params)

                for depth in depths:
                    depth = int(depth)

                    for mask_ratio in mask_ratios:
                        mask_ratio = float(mask_ratio)

                        # run name / directories
                        run_name = make_run_name(
                            corr_type=corr_type,
                            depth=depth,
                            mask_ratio=mask_ratio,
                            seed=seed,
                            patch_size=PRE.patch_size,
                            embed_dim=PRE.embed_dim,
                            num_heads=PRE.num_heads,
                            corr_param_tag=corr_tag,
                        )

                        pre_out = PRE.sweep_out / run_name
                        pre_out.mkdir(parents=True, exist_ok=True)

                        if pre_train:
                            # -------------------------
                            # (1) PRETRAIN
                            # -------------------------
                            pre_cmd = build_pretrain_cmd(
                                S=PRE,
                                seed=seed,
                                corr_type=corr_type,
                                corr_params=corr_params,
                                depth=depth,
                                mask_ratio=mask_ratio,
                                out_dir=pre_out,
                            )

                            print(f"[PRE] seed={seed} corr={corr_type} params={corr_tag} d={depth} r={mask_ratio}")
                            if corr_type in ("gauss", "gaussian"):
                                print(f"      [GAUSS] {corr_params}")
                            rc = run_cmd(pre_cmd, cwd=PRE.root)
                            if rc != 0:
                                failures.append(f"PREFAIL: {run_name} (rc={rc})")
                                continue

                        # -------------------------
                        # (2) FINETUNE (for each chosen ckpt spec)
                        # -------------------------
                        for ckpt_mode, ckpt_epoch in ckpt_specs:
                            ckpt = find_checkpoint(
                                pre_out,
                                tag=PRE.tag,
                                mode=str(ckpt_mode),
                                epoch=ckpt_epoch,
                            )
                            if ckpt is None or (not ckpt.is_file()):
                                failures.append(f"NOCKPT: {run_name} mode={ckpt_mode} epoch={ckpt_epoch}")
                                continue

                            ft_suffix = "__HEADONLY" if bool(head_only) else ""
                            ckpt_tag = (
                                f"__from_epoch_{ckpt_epoch}"
                                if (ckpt_mode == "epoch" and ckpt_epoch is not None)
                                else ("__last" if ckpt_mode == "last" else "")
                            )
                            ft_out = FT.finetune_out / f"{run_name}__FT_ep{FT.finetune_epochs}{ckpt_tag}{ft_suffix}"
                            ft_out.mkdir(parents=True, exist_ok=True)

                            ft_cmd = build_finetune_cmd(
                                S=FT,
                                seed=seed,
                                corr_type=corr_type,
                                corr_params=corr_params,
                                depth=depth,
                                pretrained_ckpt=ckpt,
                                out_dir=ft_out,
                                head_only=bool(head_only),
                            )

                            print(
                                f"[FT ] seed={seed} corr={corr_type} params={corr_tag} "
                                f"d={depth} r={mask_ratio} ckpt={ckpt.name} "
                                f"(mode={ckpt_mode} epoch={ckpt_epoch}) pooling={pooling}"
                            )
                            rc2 = run_cmd(ft_cmd, cwd=FT.root)
                            if rc2 != 0:
                                failures.append(f"FTFAIL: {run_name} mode={ckpt_mode} epoch={ckpt_epoch} (rc={rc2})")
                                continue

    if failures:
        print("\n[!] Some runs failed:")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)

    print("\n[✓] All runs completed.")
