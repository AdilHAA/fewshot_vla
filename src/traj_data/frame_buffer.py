"""Rolling opening-clip buffer for eval-time live conditioning (n_obs_steps=1).

Anchored at the first frame seen (causal); pure torch.
"""
from __future__ import annotations

import torch

from .augment import opening_clip_indices


class FrameBuffer:
    def __init__(self, num_frames: int, stride: int = 1):
        self.num_frames = num_frames
        self.stride = stride
        self._frames: list = []

    def reset(self) -> None:
        self._frames = []

    @property
    def ready(self) -> bool:
        return len(self._frames) > 0

    def push(self, frame: torch.Tensor) -> None:
        self._frames.append(frame)

    def clip(self) -> torch.Tensor:
        if not self._frames:
            raise RuntimeError("FrameBuffer empty")
        idx = opening_clip_indices(len(self._frames), self.num_frames, self.stride)
        return torch.stack([self._frames[i] for i in idx])
