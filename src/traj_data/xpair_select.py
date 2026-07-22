"""Conditioning selectors: p_self train pairing over the within-task cartesian
product of trajectories, and the deterministic eval pick. Lerobot-free."""
from __future__ import annotations

import torch


def _choice(n: int, g: torch.Generator) -> int:
    return int(torch.randint(n, (1,), generator=g).item())


def pack_conditioning(samples):
    """samples: per-batch-item lists of (L_i, D) demo tensors -> padded batch.
    Returns (traj (B,Lmax,D), mask (B,Lmax) True==pad, marks (B,Lmax) long with
    1 on each demo's first token, 2 on its last, 0 elsewhere)."""
    B = len(samples)
    lens = [sum(d.shape[0] for d in demos) for demos in samples]
    L, D = max(lens), samples[0][0].shape[-1]
    traj = samples[0][0].new_zeros(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    marks = torch.zeros(B, L, dtype=torch.long)
    for b, demos in enumerate(samples):
        off = 0
        for d in demos:
            n = d.shape[0]
            traj[b, off:off + n] = d
            marks[b, off] = 1
            marks[b, off + n - 1] = 2
            off += n
        mask[b, :off] = False
    return traj, mask, marks


def select_train_conditioning(cache, episode_idx, task_idx, p_self: float,
                              k: int, generator: torch.Generator):
    """Per sample, Bernoulli(p_self): with prob p_self the context is the imitated
    episode's own ORIGINAL demo (the diagonal of the within-task cartesian product);
    otherwise k (=1 in the clean grid) original demos of OTHER episodes of the task
    (off-diagonal), resampled every call. Single-episode tasks fall back to self."""
    samples = []
    for b in range(len(episode_idx)):
        ep, t = int(episode_idx[b]), int(task_idx[b])
        rows = cache.rows_of_task(t)                       # originals only
        own = [r for r in rows if cache.records[r]["episode"] == ep] or rows
        other = [r for r in rows if cache.records[r]["episode"] != ep]
        use_self = (not other) or p_self >= 1.0 or (
            p_self > 0.0 and torch.rand(1, generator=generator).item() < p_self)
        if use_self:
            sel = [own[_choice(len(own), generator)]]
        else:
            k_step = min(k, len(other))
            perm = torch.randperm(len(other), generator=generator).tolist()
            sel = [other[i] for i in perm[:k_step]]
        samples.append([cache.read_row(r) for r in sel])
    return pack_conditioning(samples)


def select_eval_conditioning(cache, task_index: int, k: int, seed: int):
    """Deterministic k original demos of the task (seed = seed + task_index).
    Returns a list of (L,D) tensors; the caller packs across envs."""
    rows = cache.rows_of_task(task_index)                  # originals only
    if not rows:
        raise KeyError(f"no cached original demos for task {task_index}")
    g = torch.Generator().manual_seed(int(seed) + int(task_index))
    perm = torch.randperm(len(rows), generator=g).tolist()
    return [cache.read_row(rows[i]) for i in perm[:k]]
