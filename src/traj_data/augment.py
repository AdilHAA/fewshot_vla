"""Trajectory-frame selection and appearance augmentation for the offline cache."""
from __future__ import annotations

import torch


def opening_clip_indices(n_avail: int, num_frames: int, stride: int) -> list[int]:
    """Forward indices from the episode start, clamped (pad by repeating last)."""
    return [min(i * stride, n_avail - 1) for i in range(num_frames)]


def color_jitter(frames: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic per-channel gain/bias jitter. frames (T,C,H,W) in [0,1].

    gain in [0.7,1.3], bias in [-0.1,0.1], applied per RGB channel.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    gain = (0.7 + 0.6 * torch.rand(3, generator=g)).view(1, 3, 1, 1)
    bias = (-0.1 + 0.2 * torch.rand(3, generator=g)).view(1, 3, 1, 1)
    return (frames * gain + bias).clamp(0.0, 1.0)
