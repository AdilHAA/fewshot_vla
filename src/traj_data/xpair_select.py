"""Cross-pair conditioning selectors for train and eval.

Train: per-sample self/same-task selection from a TrajCache (color variants
enter via random variant choice). Eval: retrieval with a kill-switch fallback
to the self clip. All logic is pure torch + cache.
"""
from __future__ import annotations

import torch

from .retrieval import build_key


def _variants_by_episode(cache) -> dict:
    m: dict = {}
    for r in cache.records:
        m.setdefault(r["episode"], []).append(r["variant"])
    return m


def _choice(n: int, generator) -> int:
    return int(torch.randint(n, (1,), generator=generator)[0]) if n > 1 else 0


def select_train_conditioning(cache, episode_idx, task_idx, p_self: float, generator):
    """Per-sample self/same-task clip selection.

    episode_idx, task_idx: length-B int sequences (the imitated episode + its task).
    Returns (traj (B, T*Ntok, D_enc), mask=None) — equal-length clips, no padding.
    """
    variants = _variants_by_episode(cache)
    clips = []
    for b in range(len(episode_idx)):
        ea, t = int(episode_idx[b]), int(task_idx[b])
        use_self = torch.rand(1, generator=generator).item() < p_self
        if use_self:
            ep = ea
        else:
            cands = [e for e in cache.episodes_of_task(t) if e != ea] or [ea]
            ep = cands[_choice(len(cands), generator)]
        vs = variants.get(ep, [0])
        clips.append(cache.read(ep, vs[_choice(len(vs), generator)]))
    return torch.stack(clips), None


def select_eval_conditioning(cache, query_clip, frame_emb, text_emb, k: int, tau: float,
                             query_task_index: int, beta_t: float, beta_f: float):
    """Retrieval conditioning with kill-switch to the self clip.

    query_clip: (T*Ntok, D_enc) live opening clip (also the self fallback).
    frame_emb: (D_frame,) pooled query frame emb; text_emb: (D_text,).
    Returns (traj (1, L, D_enc), mask=None, provenance: list[str], used_fallback: bool).
    """
    query = build_key(text_emb, frame_emb, beta_t, beta_f)
    res = cache.retrieve(query, k=k, tau=tau, query_task_index=query_task_index)
    if res.kill_switch:
        return query_clip.unsqueeze(0), None, ["self_fallback"], True
    clips = cache.read_rows(res.rows)                     # (k, T*Ntok, D)
    traj = clips.reshape(1, -1, clips.shape[-1])          # (1, k*T*Ntok, D)
    return traj, None, res.provenance, False
