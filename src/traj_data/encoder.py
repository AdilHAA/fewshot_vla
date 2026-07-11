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

ENCODER_FORMAT = {"dino": "cls", "vjepa2": "tubelet_meanpool"}


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
def vjepa2_encode(model, mean, std, frames: torch.Tensor) -> torch.Tensor:
    """frames: (B,T,C,H,W), T even -> (B, T//2, D) fp16 — mean of the spatial tokens
    inside each tubelet (V-JEPA2 has no CLS; this is the per-tubelet analog)."""
    b, t, c, h, w = frames.shape
    assert t % 2 == 0, "vjepa2 needs an even frame count (tubelet_size=2)"
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(VJEPA2_SIZE, VJEPA2_SIZE), mode="bilinear", align_corners=False)
    x = ((x - mean) / std).reshape(b, t, c, VJEPA2_SIZE, VJEPA2_SIZE)
    model_dtype = next(model.parameters()).dtype
    tok = model.get_vision_features(x.to(model_dtype))    # (B, (T//2)*n_sp, D)
    n_tub = t // 2
    n_sp = tok.shape[1] // n_tub
    return tok.reshape(b, n_tub, n_sp, -1).mean(dim=2).to(torch.float16)


_ENCODE_FN = {"dino": dino_encode, "vjepa2": vjepa2_encode}
_DEFAULT_MODEL_ID = {"dino": "facebook/dinov2-base", "vjepa2": "facebook/vjepa2-vitl-fpc64-256"}


def build_traj_encoder(encoder_type: str, model_id: str | None = None,
                       device="cpu", dtype=torch.float32):
    """Load a frozen clip encoder and return (model, encode_fn). encode_fn maps
    frames (B,T,C,H,W) in [0,1] -> (B, N, D) fp16. Shared by the offline builder
    and the live eval path so preprocessing is identical."""
    from transformers import AutoModel

    if encoder_type not in _ENCODE_FN:
        raise ValueError(f"unknown encoder_type {encoder_type!r}; use 'dino' or 'vjepa2'")
    model = AutoModel.from_pretrained(model_id or _DEFAULT_MODEL_ID[encoder_type])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    mean, std = imagenet_buffers(device, dtype)
    enc = _ENCODE_FN[encoder_type]
    return model, (lambda frames: enc(model, mean, std, frames.to(device)))
