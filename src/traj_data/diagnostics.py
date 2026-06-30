"""Measure-only diagnostics for trajectory conditioning (no grad, not in loss)."""
from __future__ import annotations

import torch


def flat_lora_norm(weights: dict) -> float:
    parts = [t.reshape(-1) for layers in weights.values()
             for pair in layers.values() for t in pair]
    return float(torch.cat(parts).float().norm().item()) if parts else 0.0


def action_delta(a1: torch.Tensor, a2: torch.Tensor) -> float:
    d = (a1 - a2).reshape(a1.shape[0], -1).float()
    return float(d.norm(dim=-1).mean().item())


def z_shuffle(z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(z.shape[0], generator=generator)
    return z[perm]
