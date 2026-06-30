"""Retrieval-key construction and brute-force cosine top-k (no faiss)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_key(text_emb, frame_emb, beta_t: float, beta_f: float):
    """Per-block L2-normalize, weight, concat. Supports (..,D) batched inputs."""
    t = F.normalize(text_emb, dim=-1) * beta_t
    f = F.normalize(frame_emb, dim=-1) * beta_f
    return torch.cat([t, f], dim=-1)


def cosine_topk(query, keys, k: int, tau: float):
    """query (Dk,), keys (N,Dk) -> (top_idx, top_sims, kill_switch)."""
    if keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, torch.empty(0), True
    # Cast to float32: keys.pt is stored fp16 but a live query may be fp32, and
    # CPU fp16 matmul is unsupported on some builds.
    q = F.normalize(query.float(), dim=-1)
    kk = F.normalize(keys.float(), dim=-1)
    sims = kk @ q                                   # (N,)
    k_eff = min(k, sims.numel())
    top_sims, top_idx = torch.topk(sims, k_eff)
    kill = bool(top_sims.numel() == 0 or top_sims[0].item() < tau)
    return top_idx, top_sims, kill
