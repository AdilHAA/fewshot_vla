"""Frozen-encoder helper: trajectory frames -> token latents.

Used by both the offline cache builder and the live eval path so the
resize/normalize/dtype pipeline stays identical in both contexts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_buffers(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


@torch.no_grad()
def dino_encode(model, mean, std, frames: torch.Tensor) -> torch.Tensor:
    """frames: (B,T,C,H,W) in [0,1] -> (B, T*Ntok, D) fp16, frozen/no-grad.

    Frames are concatenated along the token axis; CLS token is kept.
    """
    b, t, c, h, w = frames.shape
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    model_dtype = next(model.parameters()).dtype
    out = model(pixel_values=x.to(model_dtype))
    tok = out.last_hidden_state                      # (B*T, Ntok, D)
    ntok, d = tok.shape[1], tok.shape[2]
    return tok.reshape(b, t * ntok, d).to(torch.float16)
