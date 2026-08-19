"""Uniform temporal striding for conditioning clips.

Truncating a demo to its first frames throws away most of the trajectory; a uniform
stride keeps the full temporal extent (start -> end) at a lower frame rate. ONE index
function drives BOTH the Qwen3.5 video-token cache build and the scratch-control
derivation from the dino CLS cache, so the two arms see bit-identical frame sets.

Lerobot-free, torch-free.
"""
from __future__ import annotations

import numpy as np


def stride_indices(n_frames: int, budget: int) -> np.ndarray:
    """Frame indices covering the WHOLE clip in `budget` (even) samples.

    linspace endpoints included (frame 0 and the last frame always selected), so the
    stride grid is deterministic and identical for every consumer. An even budget
    matters for video towers with temporal_patch_size=2 (frames are consumed in
    pairs); odd budgets are rounded up.
    """
    budget = max(int(budget), 2)
    budget += budget % 2
    T = int(n_frames)
    if T <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    if T <= budget:
        return np.arange(T, dtype=np.int64)
    return np.unique(np.linspace(0, T - 1, budget).round().astype(np.int64))


def stride_slice(tokens_per_frame: np.ndarray, budget: int) -> np.ndarray:
    """Apply the stride to a per-frame token array (e.g. dino CLS, one row/frame).

    Selects the same frame indices `stride_indices` would — the scratch control thus
    conditions on exactly the frames the Qwen video cache encodes."""
    return tokens_per_frame[stride_indices(len(tokens_per_frame), budget)]
