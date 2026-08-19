"""Uniform temporal striding for conditioning clips: every k-th frame.

A FIXED FRAME BUDGET (keep N frames of any clip) silently conditions long episodes
at a lower resolution than short ones — a 505-frame demo would lose 16x more
temporal information than a 75-frame one. A fixed INTERVAL keeps the resolution
identical across episodes (every k-th frame, first to last included), so the
conditioning quality does not depend on episode length. Token count then scales
with length, exactly like the dino/vjepa full-clip arms.

ONE index function drives BOTH the Qwen3.5 video-token cache build and the
scratch-control derivation from the dino CLS cache, so the two arms see
bit-identical frame sets.

Lerobot-free, torch-free.
"""
from __future__ import annotations

import numpy as np


def stride_indices(n_frames: int, every: int) -> np.ndarray:
    """Frame indices 0, every, 2*every, ... over the WHOLE clip (endpoints kept).

    The last frame is always included (append it when the grid stops short), so the
    full temporal extent is covered regardless of length. The count is forced EVEN:
    video towers with temporal_patch_size=2 consume frames in pairs. `every` <= 1
    keeps every frame.
    """
    T, k = int(n_frames), max(int(every), 1)
    if T <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    idx = list(range(0, T, k))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    if len(idx) % 2:                                   # odd -> drop the second-to-last
        if len(idx) > 2:
            idx.pop(-2)
        else:                                          # 2 samples, odd impossible; keep guard
            idx = idx[:2] if len(idx) >= 2 else idx + [T - 1]
    return np.array(sorted(set(idx)), dtype=np.int64)


def stride_slice(tokens_per_frame: np.ndarray, every: int) -> np.ndarray:
    """Apply the stride to a per-frame token array (e.g. dino CLS, one row/frame).

    Selects the same frame indices `stride_indices` would — the scratch control thus
    conditions on exactly the frames the Qwen video cache encodes."""
    return tokens_per_frame[stride_indices(len(tokens_per_frame), every)]
