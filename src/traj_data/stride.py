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


def pair_timestamps(src_len: int, every: int, fps: float) -> list[float]:
    """One timestamp (seconds) per 49-token unit, i.e. per temporal patch.

    The tower folds frames in pairs (temporal_patch_size=2), so `stride_indices`
    returns an even count and unit i covers frames (idx[2i], idx[2i+1]). Qwen3.5's
    prompt labels that unit with the MEAN of the two frame times — see
    Qwen3VLProcessor._calculate_timestamps (transformers 5.7.0,
    processing_qwen3_vl.py:257-268), which averages the timestamps inside each
    temporal patch before emitting `<{t:.1f} seconds>`. Reproducing the average here
    is what makes the cached tokens land in a prompt the trunk has actually seen —
    in the processor's ORDER of operations (divide, then average): `{t:.1f}` rounds at
    .x5 boundaries, where (a+b)/2/fps and (a/fps+b/fps)/2 differ by one ulp and print
    a different label.
    """
    idx = stride_indices(src_len, every)
    fps = float(fps)
    return [(float(idx[i]) / fps + float(idx[i + 1]) / fps) / 2.0 for i in range(0, len(idx), 2)]
