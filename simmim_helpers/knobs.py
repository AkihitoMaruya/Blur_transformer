#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:32:24 2025

@author: akihitomaruya
"""
# simmim_helpers/knobs.py
from __future__ import annotations
import torch

def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def _get_encoder(model):
    m = _unwrap(model)
    enc = getattr(m, "encoder", m)
    return _unwrap(enc)

def set_model_knobs(model, *, pos_noise_std: float, lambda_span: float, logger):
    enc = _get_encoder(model)

    set_pos = False
    set_span = False

    if hasattr(enc, "pos_noise_std"):
        enc.pos_noise_std = float(pos_noise_std)
        set_pos = True

    if hasattr(enc, "lambda_span"):
        enc.lambda_span = float(lambda_span)
        set_span = True

    logger.info(
        f"[Knobs] encoder={type(enc).__name__} "
        f"pos_noise_std={float(pos_noise_std)} -> {'OK' if set_pos else 'MISSING'} | "
        f"lambda_span={float(lambda_span)} -> {'OK' if set_span else 'MISSING'}"
    )

    if hasattr(enc, "blocks") and len(enc.blocks) > 0 and hasattr(enc.blocks[0], "attn"):
        a0 = enc.blocks[0].attn
        logger.info(
            f"[Knobs] attn0={type(a0).__name__} "
            f"has_record_attn={hasattr(a0,'record_attn')} has_last_attn={hasattr(a0,'last_attn')}"
        )

def get_attention_span_loss_raw(model) -> torch.Tensor:
    """
    RAW span loss from encoder.attention_span_loss() if present; else 0.
    """
    m = _unwrap(model)
    enc = _get_encoder(model)

    fn = getattr(enc, "attention_span_loss", None)
    if callable(fn):
        out = fn()
        if torch.is_tensor(out):
            return out
        device = next(m.parameters()).device
        return torch.tensor(float(out), device=device)

    device = next(m.parameters()).device
    return torch.zeros((), device=device)

def get_attention_span_loss(model) -> torch.Tensor:
    """
    WEIGHTED span loss: encoder.lambda_span * RAW.
    If encoder has no lambda_span attribute, defaults to 0 (i.e., disables).
    """
    m = _unwrap(model)
    enc = _get_encoder(model)

    raw = get_attention_span_loss_raw(model)

    lam = getattr(enc, "lambda_span", 0.0)
    try:
        lam_f = float(lam)
    except Exception:
        lam_f = 0.0

    if lam_f == 0.0:
        # keep correct device/dtype
        return raw.detach() * 0.0

    return raw * lam_f
