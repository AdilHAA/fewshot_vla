"""Shared frozen-encoder helper: trajectory frames -> token latents.

Used by BOTH the offline cache builder and (in P3) the eval live path, so the
exact resize/normalize/dtype pipeline is identical offline and online. The
parent policy's single-frame `_dino_features` is a separate legacy path and is
NOT touched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def encoder_format(encoder_type: str, vjepa_grid: int = 2) -> str:
    """Cache-header format tag: dino = one CLS per frame; vjepa2 = grid-pooled
    tubelet tokens (grid=1 is the legacy full mean-pool)."""
    if encoder_type == "dino":
        return "cls"
    if encoder_type == "vjepa2":
        return f"tubelet_grid{int(vjepa_grid)}"
    raise ValueError(f"unknown encoder_type {encoder_type!r}")


def imagenet_buffers(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


@torch.no_grad()
def dino_encode(model, mean, std, frames: torch.Tensor) -> torch.Tensor:
    """frames: (B,T,C,H,W) in [0,1] -> (B, T, D) fp16 — the CLS token of every frame."""
    b, t, c, h, w = frames.shape
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    model_dtype = next(model.parameters()).dtype
    out = model(pixel_values=x.to(model_dtype))
    cls = out.last_hidden_state[:, 0]                     # (B*T, D)
    return cls.reshape(b, t, -1).to(torch.float16)


VJEPA2_SIZE = 256


@torch.no_grad()
def vjepa2_encode(model, mean, std, frames: torch.Tensor, grid: int = 2) -> torch.Tensor:
    """frames: (B,T,C,H,W), T even -> (B, (T//2)·grid², D) fp16.

    V-JEPA2 has no CLS; each tubelet's 16×16 spatial map is average-pooled into a
    grid×grid block grid (grid=1 reproduces the legacy full mean-pool). Tokens are
    ordered tubelet-major, row-major within the grid."""
    b, t, c, h, w = frames.shape
    assert t % 2 == 0, "vjepa2 needs an even frame count (tubelet_size=2)"
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(VJEPA2_SIZE, VJEPA2_SIZE), mode="bilinear", align_corners=False)
    x = ((x - mean) / std).reshape(b, t, c, VJEPA2_SIZE, VJEPA2_SIZE)
    model_dtype = next(model.parameters()).dtype
    tok = model.get_vision_features(x.to(model_dtype))    # (B, (T//2)*n_sp, D)
    n_tub = t // 2
    n_sp = tok.shape[1] // n_tub
    side = int(n_sp ** 0.5)
    assert side * side == n_sp and side % grid == 0, \
        f"spatial map {n_sp} not divisible into a {grid}×{grid} grid"
    tok = tok.reshape(b, n_tub, grid, side // grid, grid, side // grid, -1)
    tok = tok.mean(dim=(3, 5))                            # (B, n_tub, grid, grid, D)
    return tok.reshape(b, n_tub * grid * grid, -1).to(torch.float16)


_DEFAULT_MODEL_ID = {"dino": "facebook/dinov2-base", "vjepa2": "facebook/vjepa2-vitl-fpc64-256"}


def build_traj_encoder(encoder_type: str, model_id: str | None = None,
                       device="cpu", dtype=torch.float32, vjepa_grid: int = 2):
    """Load a frozen clip encoder and return (model, encode_fn). encode_fn maps
    frames (B,T,C,H,W) in [0,1] -> (B, N, D) fp16. Shared by the offline builder
    and the live eval path so preprocessing is identical."""
    from transformers import AutoModel

    if encoder_type not in _DEFAULT_MODEL_ID:
        raise ValueError(f"unknown encoder_type {encoder_type!r}; use 'dino' or 'vjepa2'")
    model = AutoModel.from_pretrained(model_id or _DEFAULT_MODEL_ID[encoder_type])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    mean, std = imagenet_buffers(device, dtype)
    if encoder_type == "dino":
        return model, (lambda frames: dino_encode(model, mean, std, frames.to(device)))
    return model, (lambda frames: vjepa2_encode(model, mean, std, frames.to(device),
                                                grid=vjepa_grid))
