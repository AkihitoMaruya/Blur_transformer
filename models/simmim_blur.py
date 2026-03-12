# =========================================================
# models/simmim_blur.py
# =========================================================
"""
models/simmim_blur.py
--------------------------------------------------------
SimMIM + PIXEL CORRUPTION (blur or blank) computed ONLY here.
Encoder NEVER applies token-level corruption.

Based on SimMIM (c) 2021 Microsoft (MIT)
Written/modified by Akihito Maruya

Fixes:
  - MPS-safe contiguity before reshape/conv
  - CIFAR/VIT patch-size fix: decoder upsample stride must match VIT patch size
    (encoder_stride = model_patch for ViT), not hard-coded 16.
  - FIX: recon visualization saturation:
      * return x_rec01 (pixel-space [0,1]) when return_inputs01=True
      * use mean/std buffers so norm/unnorm are consistent with dataset
  - NEW: dataset-aware mean/std (ImageNet vs CIFAR10) controlled by config.DATA.DATASET
--------------------------------------------------------
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer

from .patchwise_augment import (
    PatchwisePyrBlurFromMask,
    PatchwiseBlankFromMask,
    build_pyr_blur_from_config,
    build_blank_from_config,
)

# -------------------------
# Dataset normalization stats
# -------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# CIFAR-10 channel stats (standard)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  =  (0.2470, 0.2430, 0.2610)


def _pick_norm_stats_from_config(config) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ds = str(getattr(config.DATA, "DATASET", "imagenet")).lower().strip()
    if ds == "cifar10":
        return _CIFAR10_MEAN, _CIFAR10_STD
    # default
    return _IMAGENET_MEAN, _IMAGENET_STD


# -------------------------
# Mask grid conversion
# -------------------------
def _mask_to_patch_grid(
    mask_model: torch.Tensor,
    *,
    img_size: int,
    model_patch: int,
    out_patch: int,
) -> torch.Tensor:
    """
    Convert SimMIM token mask (model_patch grid) -> out_patch grid.

    mask_model: (B,Hm,Wm) or (B,1,Hm,Wm) or (Hm,Wm)
      where Hm = img_size/model_patch

    returns: (B,Ho,Wo) boolean
      where Ho = img_size/out_patch

    Rule: out patch is True if ANY underlying model-patch is True (max-pool in patch space).
    """
    if mask_model.dim() == 2:
        mask_model = mask_model.unsqueeze(0)
    if mask_model.dim() == 4 and mask_model.shape[1] == 1:
        mask_model = mask_model.squeeze(1)
    if mask_model.dim() != 3:
        raise ValueError(
            f"mask_model must be (B,Hm,Wm) or (B,1,Hm,Wm) or (Hm,Wm), got {tuple(mask_model.shape)}"
        )

    B, Hm, Wm = mask_model.shape
    exp = img_size // model_patch
    if (Hm != exp) or (Wm != exp):
        raise ValueError(f"mask_model grid {Hm}x{Wm} must be {exp}x{exp} (img_size/model_patch).")

    m = mask_model
    if m.dtype != torch.bool:
        m = m > 0.5

    if out_patch == model_patch:
        return m

    if out_patch % model_patch != 0:
        raise ValueError(f"out_patch={out_patch} must be a multiple of model_patch={model_patch}")

    s = out_patch // model_patch
    Ho = img_size // out_patch
    Wo = img_size // out_patch
    if (Hm % s) or (Wm % s):
        raise ValueError(f"Model grid {Hm}x{Wm} not divisible by s={s} (out_patch/model_patch).")

    # MPS-safe reshape
    m2 = m.reshape(B, Ho, s, Wo, s).amax(dim=4).amax(dim=2)
    return m2


def _mask_to_pixel_mask(mask_model_bool: torch.Tensor, *, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    mask_model_bool: (B,Hm,Wm) bool -> (B,1,H,W) bool by repeating patch_size.
    """
    if mask_model_bool.dim() != 3 or mask_model_bool.dtype != torch.bool:
        raise ValueError(
            f"mask_model_bool must be (B,Hm,Wm) bool, got {tuple(mask_model_bool.shape)} {mask_model_bool.dtype}"
        )

    mask_pix = mask_model_bool.unsqueeze(1)  # (B,1,Hm,Wm)
    mask_pix = mask_pix.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask_pix[..., :H, :W]

class SwinTransformerForSimMIM_NoTokenMask(SwinTransformer):
    """Same as SwinTransformer, but accepts (x, mask) and ignores mask."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

        #  knobs (so knobs.py can set them)
        self.lambda_span: float = 0.0
        self.pos_noise_std: float = 0.0
        self._span_dist2_cache = {}

    def forward(self, x, mask=None):
        x = self.patch_embed(x)

        # APPLY POS NOISE IN TRAIN + VAL (only if APE)
        if self.ape:
            pos_noise_std = float(getattr(self, "pos_noise_std", 0.0))
            if pos_noise_std > 0.0:
                noise = torch.randn_like(self.absolute_pos_embed) * pos_noise_std
                x = x + (self.absolute_pos_embed + noise)
            else:
                x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        # enable attention recording when span enabled
        lam = float(getattr(self, "lambda_span", 0.0))
        record = (lam > 0.0)
        for layer in self.layers:
            for blk in layer.blocks:
                attn = getattr(blk, "attn", None)
                if attn is not None and hasattr(attn, "record_attn"):
                    attn.record_attn = record

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2).contiguous()
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x
class VisionTransformerForSimMIM_NoTokenMask(VisionTransformer):
    """Same as VisionTransformer, but accepts (x, mask) and ignores mask."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # APPLY POS NOISE IN TRAIN + VAL
        if self.pos_embed is not None:
            pos_noise_std = float(getattr(self, "pos_noise_std", 0.0))
            if pos_noise_std > 0.0:
                noise = torch.randn_like(self.pos_embed) * pos_noise_std
                x = x + (self.pos_embed + noise)
            else:
                x = x + self.pos_embed

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        # enable attention recording when span enabled
        lam = float(getattr(self, "lambda_span", 0.0))
        record = (lam > 0.0)
        for blk in self.blocks:
            blk.attn.record_attn = record

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x



# -------------------------
# Main SimMIM module: pixel corruption only
# -------------------------
class SimMIMBlur(nn.Module):
    """
    Pixel corruption is computed ONLY here:
      - CORRUPTION="blur"  -> pyramid blur on masked patches or gaussian with high-sf cutoff
      - CORRUPTION="blank" -> constant-fill blank on masked patches

    Encoder NEVER applies token-masking.

    Inputs:
      x    : normalized (B,3,H,W) using mean/std buffers (ImageNet or CIFAR10)
      mask : SimMIM token mask at model_patch grid (B,Hm,Wm)
    """
    def __init__(
        self,
        encoder: nn.Module,
        encoder_stride: int,
        *,
        img_size: int,
        model_patch: int,
        corruption: str,  # "blur" or "blank"
        corr_patch: int,  # usually MASK_PATCH_SIZE
        blur_module: Optional[PatchwisePyrBlurFromMask],
        blank_module: PatchwiseBlankFromMask,
        loss_on_full_image: bool = False,
        mean: tuple[float, float, float] = _IMAGENET_MEAN,
        std: tuple[float, float, float] = _IMAGENET_STD,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder_stride = int(encoder_stride)

        self.img_size = int(img_size)
        self.model_patch = int(model_patch)

        self.corruption = str(corruption).lower().strip()
        if self.corruption not in ("blur", "blank"):
            raise ValueError(f"corruption must be 'blur' or 'blank', got {corruption!r}")

        self.corr_patch = int(corr_patch)
        self.blur = blur_module
        self.blank = blank_module

        self.loss_on_full_image = bool(loss_on_full_image)

        # register mean/std buffers so they move with device
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean_t, persistent=False)
        self.register_buffer("_std",  std_t,  persistent=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = int(self.encoder.in_chans)
        self.patch_size = int(self.encoder.patch_size)

        if self.corruption == "blur" and self.blur is None:
            raise ValueError("corruption='blur' but blur_module is None")

    def _unnorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std  = self._std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _norm(self, x01: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=x01.device, dtype=x01.dtype)
        std  = self._std.to(device=x01.device, dtype=x01.dtype)
        return (x01 - mean) / std

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_inputs01: bool = False,
    ):
        """
        Returns:
          - loss
          - or (loss, x_rec01, x_clean01, x_corr01) if return_inputs01=True

        Training loss is ALWAYS computed in normalized space vs clean x.
        """
        # ---- normalize mask to (B,Hm,Wm) bool ----
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.dim() != 3:
            raise ValueError(
                f"mask must be (B,Hm,Wm) or (B,1,Hm,Wm) or (Hm,Wm), got {tuple(mask.shape)}"
            )

        mask = mask.to(device=x.device, non_blocking=True)
        mask_bool = mask if mask.dtype == torch.bool else (mask > 0.5)

        # ---- clean pixels in [0,1] ----
        x_clean01 = self._unnorm(x)

        # ---- build corruption mask on corr_patch grid ----
        mask_corr_patch = _mask_to_patch_grid(
            mask_bool,
            img_size=self.img_size,
            model_patch=self.model_patch,
            out_patch=self.corr_patch,
        )  # (B,Hp,Wp) bool

        # ---- apply corruption in pixel space ----
        if self.corruption == "blur":
            x_corr01 = self.blur(x_clean01, mask_corr_patch)  # type: ignore[arg-type]
        else:
            x_corr01 = self.blank(x_clean01, mask_corr_patch)

        x_corr01 = x_corr01.to(dtype=x.dtype)

        # ---- encoder input is corrupted pixels, normalized ----
        x_corr = self._norm(x_corr01)

        # ---- encoder NEVER token-masks ----
        z = self.encoder(x_corr, mask=None)
        z = z.contiguous()

        # ---- decode to normalized image ----
        x_rec = self.decoder(z)

        # ---- loss target is ALWAYS clean normalized image ----
        x_tgt = x

        if self.loss_on_full_image:
            loss = F.l1_loss(x_tgt, x_rec, reduction="mean")
        else:
            B, _, H, W = x.shape
            mask_pix = _mask_to_pixel_mask(mask_bool, patch_size=self.patch_size, H=H, W=W)  # (B,1,H,W)
            loss_map = F.l1_loss(x_tgt, x_rec, reduction="none")
            loss = (loss_map * mask_pix).sum() / (mask_pix.sum() + 1e-5) / self.in_chans

        if return_inputs01:
            # return reconstruction in pixel space for correct viz
            x_rec01 = self._unnorm(x_rec)
            return loss, x_rec01, x_clean01, x_corr01

        return loss


# -------------------------
# Builder
# -------------------------
def build_simmim_blur(config):
    model_type = config.MODEL.TYPE

    if model_type == "swin":
        model_patch = int(config.MODEL.SWIN.PATCH_SIZE)
        encoder = SwinTransformerForSimMIM_NoTokenMask(
            img_size=config.DATA.IMG_SIZE,
            patch_size=model_patch,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        # Swin has internal patch merging => typical output stride is 32
        encoder_stride = 32

    elif model_type == "vit":
        model_patch = int(config.MODEL.VIT.PATCH_SIZE)
        encoder = VisionTransformerForSimMIM_NoTokenMask(
            img_size=config.DATA.IMG_SIZE,
            patch_size=model_patch,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
        )

        # ViT feature map is on patch grid -> PixelShuffle must upscale by patch size
        encoder_stride = model_patch

    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    corruption = str(getattr(config.DATA, "CORRUPTION", "blur")).lower().strip()
    if corruption not in ("blur", "blank"):
        raise ValueError(f"config.DATA.CORRUPTION must be 'blur' or 'blank', got {corruption!r}")

    corr_patch = int(config.DATA.MASK_PATCH_SIZE)

    blur_module = build_pyr_blur_from_config(config) if (corruption == "blur") else None
    blank_module = build_blank_from_config(config)

    loss_on_full_image = bool(getattr(config, "LOSS_ON_FULL_IMAGE", False))

    mean, std = _pick_norm_stats_from_config(config)

    model = SimMIMBlur(
        encoder=encoder,
        encoder_stride=encoder_stride,
        img_size=int(config.DATA.IMG_SIZE),
        model_patch=model_patch,
        corruption=corruption,
        corr_patch=corr_patch,
        blur_module=blur_module,
        blank_module=blank_module,
        loss_on_full_image=loss_on_full_image,
        mean=mean,
        std=std,
    )
    return model


