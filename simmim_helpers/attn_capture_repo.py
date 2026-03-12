#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:38:19 2026

@author: akihitomaruya
"""
# simmim_helpers/attn_capture_repo.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import torch


def set_record_attn(encoder: torch.nn.Module, enabled: bool) -> int:
    """
    Enable/disable attention recording across your repo encoders.

    Your code paths:
      - ViT:  encoder.blocks[*].attn.record_attn
      - Swin: encoder.layers[*].blocks[*].attn.record_attn

    Returns: number of attention modules updated.
    """
    n = 0

    # ViT-style
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "record_attn"):
                attn.record_attn = bool(enabled)
                n += 1

    # Swin-style
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for layer in layers:
            blks = getattr(layer, "blocks", None)
            if blks is None:
                continue
            for blk in blks:
                attn = getattr(blk, "attn", None)
                if attn is not None and hasattr(attn, "record_attn"):
                    attn.record_attn = bool(enabled)
                    n += 1

    return n


def clear_last_attn(encoder: torch.nn.Module) -> int:
    """
    Clears stored attn buffers (attn.last_attn=None).
    Returns: number cleared.
    """
    n = 0

    blocks = getattr(encoder, "blocks", None)
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "last_attn"):
                attn.last_attn = None
                n += 1

    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for layer in layers:
            blks = getattr(layer, "blocks", None)
            if blks is None:
                continue
            for blk in blks:
                attn = getattr(blk, "attn", None)
                if attn is not None and hasattr(attn, "last_attn"):
                    attn.last_attn = None
                    n += 1

    return n


@torch.no_grad()
def collect_last_attn(encoder: torch.nn.Module, *, require: bool = True) -> List[torch.Tensor]:
    """
    Collect stored attention maps in shallow -> deep order.

    Returns:
      attn_maps: list of [B,H,T,T] tensors (softmaxed).
    """
    out: List[torch.Tensor] = []

    # ViT shallow->deep
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            A = getattr(attn, "last_attn", None) if attn is not None else None
            if A is not None:
                out.append(A)

    # Swin shallow->deep
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for layer in layers:
            blks = getattr(layer, "blocks", None)
            if blks is None:
                continue
            for blk in blks:
                attn = getattr(blk, "attn", None)
                A = getattr(attn, "last_attn", None) if attn is not None else None
                if A is not None:
                    out.append(A)

    if require and len(out) == 0:
        raise RuntimeError(
            "No attention maps found (all last_attn are None). "
            "Your model likely does not implement attn.last_attn recording. "
            "Use hook capture (run_with_attn_capture has an automatic hook fallback)."
        )

    return out


def _find_encoder(
    model_or_encoder: torch.nn.Module,
    encoder_attr_candidates: Tuple[str, ...] = ("encoder",),
) -> torch.nn.Module:
    enc = None
    for attr in encoder_attr_candidates:
        if hasattr(model_or_encoder, attr):
            enc = getattr(model_or_encoder, attr)
            break
    return model_or_encoder if enc is None else enc


def _install_attn_drop_hooks(
    encoder: torch.nn.Module,
    bucket: List[torch.Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Fallback capture:
      - Find each block.attn.attn_drop (ViT or Swin WindowAttention)
      - Hook its *input* (softmaxed attention) and store it.

    bucket gets appended with tensors shaped like [B,H,N,N] or [B_,H,N,N].
    Returns hook handles for removal.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook_fn(_mod, inputs, _output):
        # inputs[0] is attention after softmax, before dropout
        if not inputs:
            return
        A = inputs[0]
        if torch.is_tensor(A) and A.dim() == 4:
            bucket.append(A.detach())

    # ViT blocks
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            attn_drop = getattr(attn, "attn_drop", None)
            if attn_drop is not None:
                handles.append(attn_drop.register_forward_hook(_hook_fn))

    # Swin layers/blocks
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for layer in layers:
            blks = getattr(layer, "blocks", None)
            if blks is None:
                continue
            for blk in blks:
                attn = getattr(blk, "attn", None)
                if attn is None:
                    continue
                attn_drop = getattr(attn, "attn_drop", None)
                if attn_drop is not None:
                    handles.append(attn_drop.register_forward_hook(_hook_fn))

    return handles


@torch.no_grad()
def run_with_attn_capture(
    model_or_encoder: torch.nn.Module,
    x: torch.Tensor,
    *,
    forward_kwargs: Optional[dict] = None,
    encoder_attr_candidates: Tuple[str, ...] = ("encoder",),
) -> tuple[Any, List[torch.Tensor]]:
    """
    Convenience helper:
      - finds encoder (defaults to model.encoder if present)
      - tries native recording path (record_attn/last_attn) if available
      - if nothing recorded, automatically falls back to hook capture via attn_drop

    Returns:
      y, attn_maps  where attn_maps is list[Tensor(B,H,T,T)] shallow->deep
    """
    forward_kwargs = {} if forward_kwargs is None else dict(forward_kwargs)
    enc = _find_encoder(model_or_encoder, encoder_attr_candidates=encoder_attr_candidates)

    # 1) Try native record_attn/last_attn path
    n_set = set_record_attn(enc, True)
    clear_last_attn(enc)

    y = model_or_encoder(x, **forward_kwargs)

    attn_maps: List[torch.Tensor] = []
    try:
        attn_maps = collect_last_attn(enc, require=False)
    finally:
        # tidy native flag
        set_record_attn(enc, False)

    if len(attn_maps) > 0:
        return y, attn_maps

    # 2) Hook fallback (no need for record_attn)
    bucket: List[torch.Tensor] = []
    handles = _install_attn_drop_hooks(enc, bucket=bucket)
    try:
        # rerun forward once to capture attention via hooks
        y = model_or_encoder(x, **forward_kwargs)
    finally:
        for h in handles:
            h.remove()

    if len(bucket) == 0:
        raise RuntimeError(
            "Hook capture failed: no attention tensors were observed at attn.attn_drop inputs.\n"
            "This suggests your attention modules don't expose attn_drop, or the model isn't ViT/Swin-like.\n"
            "Inspect one block.attn to confirm it has an 'attn_drop' module."
        )

    # bucket is already shallow->deep in the order we registered hooks
    return y, bucket

