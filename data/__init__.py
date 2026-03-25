from .data_simmim_blur import (
    build_loader_simmim,
    build_val_loader_simmim,
    # NEW: CIFAR10 pretrain loaders (you added these in data_simmim_blur.py)
    build_loader_simmim_cifar10,
    build_val_loader_simmim_cifar10,
)
from .data_finetune_blur import build_loader_finetune


def _dataset_mode(config) -> str:
    # config.DATA.DATASET is what you already use elsewhere ("imagenet" / "cifar10")
    return str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def _fmt3(xs):
    return "[" + ", ".join(f"{float(x):.4f}" for x in xs) + "]"

def _log_norm_hint(config, logger, where: str):
    # Best-effort: print config.DATA.MEAN/STD if present, else tell you it's hard-coded
    mean = getattr(config.DATA, "MEAN", None)
    std  = getattr(config.DATA, "STD", None)
    if mean is not None and std is not None:
        logger.info(f"[NORM:{where}] from config: MEAN={mean} STD={std}")
    else:
        logger.info(
            f"[NORM:{where}] MEAN/STD not in config -> loaders may use hard-coded values "
            f"(ImageFolder uses ImageNet: mean={_fmt3(IMAGENET_DEFAULT_MEAN)} std={_fmt3(IMAGENET_DEFAULT_STD)})"
        )

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        ds = _dataset_mode(config)
        logger.info(f"[ROUTE] is_pretrain=True DATA.DATASET={ds}")
        _log_norm_hint(config, logger, where="pretrain")

        if ds == "cifar10":
            train_frac = float(getattr(config.DATA, "CIFAR_TRAIN_FRAC", 0.8))
            download = bool(getattr(config.DATA, "CIFAR_DOWNLOAD", False))
            logger.info(f"[ROUTE] using CIFAR10 split loader (train_frac={train_frac} download={download})")
            return build_loader_simmim_cifar10(config, logger, download=download, train_frac=train_frac)

        logger.info(f"[ROUTE] using ImageFolder loader (build_loader_simmim)")
        return build_loader_simmim(config, logger)

    logger.info("[ROUTE] is_pretrain=False -> finetune loader (build_loader_finetune)")
    _log_norm_hint(config, logger, where="finetune")
    return build_loader_finetune(config, logger)
def build_val_loader(config, logger, val_path: str, is_pretrain: bool):
    if not is_pretrain:
        return None

    ds = _dataset_mode(config)
    logger.info(f"[ROUTE] is_pretrain=True (val) DATA.DATASET={ds}")
    _log_norm_hint(config, logger, where="pretrain_val")

    if ds == "cifar10":
        train_frac = float(getattr(config.DATA, "CIFAR_TRAIN_FRAC", 0.8))
        download = bool(getattr(config.DATA, "CIFAR_DOWNLOAD", False))
        logger.info(f"[ROUTE] using CIFAR10 split VAL loader (train_frac={train_frac} download={download})")
        return build_val_loader_simmim_cifar10(config, logger, download=download, train_frac=train_frac)

    if not val_path:
        logger.info("[ROUTE] ImageFolder val_path is empty -> no val loader")
        return None

    logger.info(f"[ROUTE] using ImageFolder VAL loader (build_val_loader_simmim) val_path={val_path}")
    return build_val_loader_simmim(config, logger, val_path)
