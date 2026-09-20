"""Conditioning selectors: p_self train pairing over the within-task cartesian
product of trajectories, and the deterministic eval pick. Lerobot-free.

Selection returns ROWS, not tensors: the fusion arm packs them into one padded
batch, while the trunk arm needs each demo separately (its tokens go into a chat
prompt together with the pair timestamps of its source episode). Both arms
therefore share one selection and differ only in how they read the rows.
"""
from __future__ import annotations

import zlib

import torch


def pack_conditioning(samples, tokens_per_unit: int = 1):
    """samples: per-batch-item lists of (L_i, D) demo tensors -> padded batch.

    Returns (traj (B,Lmax,D), mask (B,Lmax) True==pad, marks (B,Lmax) long with
    1 on each demo's first token, 2 on its last, 0 elsewhere, pos (B,Lmax) long =
    index of the token's UNIT within its own demo, sub (B,Lmax) long = index of the
    token within its unit).

    `pos` counts units (frames for dino, tubelets for vjepa2), not slots: tokens are
    unit-major, so the `tokens_per_unit` tokens of one frame must share one time.
    It RESETS at every demo boundary — a running arange across demos would invent a
    single continuous timeline out of unrelated trajectories.
    Padding slots keep pos=sub=0; they are masked out as attention keys, so their
    value cannot reach the output (verified: perturbing them moves it by exactly 0).
    """
    B = len(samples)
    u = max(int(tokens_per_unit), 1)
    lens = [sum(d.shape[0] for d in demos) for demos in samples]
    L, D = max(lens), samples[0][0].shape[-1]
    traj = samples[0][0].new_zeros(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    marks = torch.zeros(B, L, dtype=torch.long)
    pos = torch.zeros(B, L, dtype=torch.long)
    sub = torch.zeros(B, L, dtype=torch.long)
    for b, demos in enumerate(samples):
        off = 0
        for d in demos:
            n = d.shape[0]
            traj[b, off:off + n] = d
            marks[b, off] = 1
            if n > 1:                      # n==1 would overwrite the start mark
                marks[b, off + n - 1] = 2
            slot = torch.arange(n)
            pos[b, off:off + n] = slot // u
            sub[b, off:off + n] = slot % u
            off += n
        mask[b, :off] = False
    return traj, mask, marks, pos, sub


def select_train_rows(cache, episode_idx, p_self: float, k: int,
                      generator: torch.Generator) -> list:
    """Per sample, Bernoulli(p_self): with prob p_self the context is the imitated
    episode's own ORIGINAL demo (the diagonal of the within-task cartesian product);
    otherwise k original demos of OTHER episodes of the SAME task (off-diagonal),
    resampled every call. Single-episode tasks fall back to self.

    The task is `cache.key_of_episode(ep)` — a bddl stem, not the batch's task_index,
    which the merged dataset shares between same-sentence tasks. A missing episode
    raises: silently conditioning on another task's demos would be invisible in the
    loss and fatal to the experiment."""
    out = []
    for ep in episode_idx:
        ep = int(ep)
        rows = cache.rows_of_task(cache.key_of_episode(ep))   # originals only
        own = cache.row_of_episode(ep)
        other = [r for r in rows if cache.records[r]["episode"] != ep]
        use_self = (not other) or p_self >= 1.0 or (
            p_self > 0.0 and torch.rand(1, generator=generator).item() < p_self)
        if use_self:
            out.append([own])
        else:
            perm = torch.randperm(len(other), generator=generator).tolist()
            out.append([other[i] for i in perm[:min(k, len(other))]])
    return out


def select_eval_rows(cache, task_key: str, k: int, seed: int) -> list:
    """Deterministic k original demos of the task. The per-task seed offset is a
    CRC of the key (task_keys are strings), so two tasks never draw the same
    permutation and the pick is stable across runs and machines."""
    rows = cache.rows_of_task(task_key)                       # originals only
    if not rows:
        raise KeyError(f"no cached original demos for task {task_key!r}")
    g = torch.Generator().manual_seed(
        int(seed) + zlib.crc32(str(task_key).encode()) % 10_000_000)
    perm = torch.randperm(len(rows), generator=g).tolist()
    return [rows[i] for i in perm[:k]]


def pack_rows(cache, rows):
    """rows: per-sample lists of cache rows -> the padded fusion-arm batch."""
    samples = [[cache.read_row(r) for r in sample] for sample in rows]
    return pack_conditioning(samples, int(cache.header.get("tokens_per_unit", 1)))
