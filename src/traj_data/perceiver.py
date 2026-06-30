"""Perceiver Resampler: compress a variable-length token sequence into M learned
latents via cross-attention. Trainable; pure torch."""
from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, hidden_size, num_latents=8, depth=2,
                 num_heads=4, ffn_mult=2, max_seq_len=8192):
        super().__init__()
        self.num_latents = num_latents
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(depth))
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(depth))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size, hidden_size * ffn_mult), nn.GELU(),
                          nn.Linear(hidden_size * ffn_mult, hidden_size)) for _ in range(depth))
        self.ln_kv = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(depth))
        self.ln_q = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(depth))
        self.ln_s = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(depth))
        self.ln_f = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(depth))

    def forward(self, tokens, key_padding_mask=None):
        b, L, _ = tokens.shape
        pos = self.pos_emb(torch.arange(L, device=tokens.device))
        x = self.input_proj(tokens) + pos.unsqueeze(0)          # (B,L,H)
        z = self.latents.unsqueeze(0).expand(b, -1, -1)         # (B,M,H)
        for i in range(len(self.cross_attn)):
            kv = self.ln_kv[i](x)
            q = self.ln_q[i](z)
            a, _ = self.cross_attn[i](q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
            z = z + a
            s, _ = self.self_attn[i](self.ln_s[i](z), self.ln_s[i](z), self.ln_s[i](z), need_weights=False)
            z = z + s
            z = z + self.ffn[i](self.ln_f[i](z))
        return z
