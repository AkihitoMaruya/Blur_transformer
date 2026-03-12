#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 18:43:22 2025

@author: akihitomaruya
"""
# models/__init__.py
from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .simmim_blur import build_simmim_blur  # handles BOTH blur and blank


def build_model(config, is_pretrain: bool = True):
    if is_pretrain:
        # CORRUPTION drives which pretrain model to build.
        # Supported: "none" (original SimMIM token-masking), "blur", "blank"
        corr = str(getattr(config.DATA, "CORRUPTION", "none")).lower().strip()
        
        if corr in ("blur", "blank"):
            return build_simmim_blur(config)

        raise ValueError(
            f"Unknown DATA.CORRUPTION={corr!r}. Use 'none', 'blur', or 'blank'."
        )

    model_type = config.MODEL.TYPE
    if model_type == "swin":
        return build_swin(config)
    if model_type == "vit":
        return build_vit(config)
    raise NotImplementedError(f"Unknown fine-tune model: {model_type}")
