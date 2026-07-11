"""FusionHyperNetwork — a backward-compatible subclass of the base HyperNetwork that
fuses extra conditioning streams (raw per-timestep trajectory tokens, per-stream type
embeddings) into LoRA generation.

Backward-compat: every fusion feature is default-OFF. When none is active, `forward`
delegates to the parent `HyperNetwork.forward` and the module has NO extra parameters,
so its state_dict is identical to the parent's (`tests/test_traj_backcompat.py`).

Active (Stage>=2) path: when `use_traj`, the raw frozen-encoder trajectory tokens
`traj_embeds` are projected (`traj_proj`: traj_dim -> hidden) straight into the concat
self-attention as a 4th stream in the tf sequence `[layer | text | (vision) | traj]`.
Demo boundaries are injected as additive mark embeddings (`traj_mark_embedding`:
1=demo-start, 2=demo-end, 0=interior). The layer tokens are read back and decoded to
`(W_down, W_up)` exactly as in the parent. Requires `encoder_type='tf'`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hyper_lora.hypernetwork import HyperNetwork


class FusionHyperNetwork(HyperNetwork):
    def __init__(
        self,
        *args,
        use_traj: bool = False,
        traj_dim: int = 0,
        stream_type_emb: bool = False,
        per_stream_null: bool = False,
        readout: str = "queries",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # base HyperNetwork; lines unchanged
        self.use_traj = use_traj
        self.stream_type_emb = stream_type_emb
        self.per_stream_null = per_stream_null
        self.readout = readout
        # Extra params are constructed ONLY behind their flag, so an all-OFF instance
        # has exactly the parent's parameter/state_dict keys (byte-compat).
        if use_traj:
            # Raw per-timestep tokens go straight into the concat self-attention;
            # boundaries between demos are additive mark embeddings (1=start, 2=end).
            self.traj_proj = nn.Linear(traj_dim, self.hidden_size)
            self.traj_mark_embedding = nn.Embedding(3, self.hidden_size)
        if stream_type_emb:
            self.stream_type_embedding = nn.Embedding(4, self.hidden_size)

    def _new_active(self) -> bool:
        return bool(
            self.use_traj
            or self.stream_type_emb
            or self.per_stream_null
            or self.readout == "xattn"
        )

    def forward(
        self,
        text_token_embeds,
        attention_mask=None,
        vlm_vision_embeds=None,
        dino_embeds=None,
        traj_embeds=None,
        traj_mask=None,
        traj_marks=None,
        **kwargs,
    ):
        # Fast-path: with no fusion feature active this is numerically and structurally
        # identical to the parent.
        if not self._new_active():
            return super().forward(
                text_token_embeds, attention_mask, vlm_vision_embeds, dino_embeds
            )
        if self.encoder_type != "tf":
            raise NotImplementedError(
                "FusionHyperNetwork trajectory stream requires encoder_type='tf'."
            )

        B = text_token_embeds.shape[0]
        device = text_token_embeds.device
        param_dtype = self.text_proj.weight.dtype

        text_h = self.text_proj(text_token_embeds.to(param_dtype))
        layer_idx_t = torch.arange(self.num_layers, device=device)
        layer_h = self.layer_embedding(layer_idx_t).unsqueeze(0).expand(B, -1, -1)

        # Context sequence: [layer | text | vision... | traj]. Layer tokens are the
        # queries read back; everything else is context. pad mask: True == ignore.
        seq_parts = [layer_h, text_h]
        pad_parts = [torch.zeros(B, self.num_layers, device=device, dtype=torch.bool)]
        stream_ids = [0] * self.num_layers + [1] * text_h.shape[1]
        if attention_mask is not None:
            pad_parts.append(~attention_mask.bool())
        else:
            pad_parts.append(torch.zeros(B, text_h.shape[1], device=device, dtype=torch.bool))

        for v in self._proj_vision(vlm_vision_embeds, dino_embeds, param_dtype):
            seq_parts.append(v)
            pad_parts.append(torch.zeros(B, v.shape[1], device=device, dtype=torch.bool))
            stream_ids += [2] * v.shape[1]

        if self.use_traj and traj_embeds is not None:
            tr = self.traj_proj(traj_embeds.to(param_dtype))
            if traj_marks is not None:
                tr = tr + self.traj_mark_embedding(traj_marks.to(device))
            seq_parts.append(tr)
            if traj_mask is not None:
                pad_parts.append(traj_mask.to(device))
            else:
                pad_parts.append(torch.zeros(B, tr.shape[1], device=device, dtype=torch.bool))
            stream_ids += [3] * tr.shape[1]

        seq = torch.cat(seq_parts, dim=1)
        src_key_padding_mask = torch.cat(pad_parts, dim=1)

        if self.stream_type_emb:
            ids = torch.tensor(stream_ids, device=device)
            seq = seq + self.stream_type_embedding(ids).unsqueeze(0)

        encoded = self.context_encoder(seq, src_key_padding_mask=src_key_padding_mask)
        layer_ctx = self.context_dropout(encoded[:, : self.num_layers, :])
        return self._emit_lora(layer_ctx, B)
