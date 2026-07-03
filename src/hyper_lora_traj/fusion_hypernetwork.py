"""FusionHyperNetwork — backward-compatible HyperNetwork subclass that fuses extra
conditioning streams (trajectory-clip latents via a Perceiver resampler, per-stream
type embeddings) into LoRA generation. With all flags off, forward delegates to the
parent and the module has no extra parameters.
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
        perceiver_latents: int = 8,
        perceiver_depth: int = 2,
        perceiver_heads: int = 4,
        perceiver_max_seq_len: int = 8192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_traj = use_traj
        self.stream_type_emb = stream_type_emb
        self.per_stream_null = per_stream_null
        self.readout = readout
        # Extra params constructed only behind their flag — an all-OFF instance has
        # exactly the parent's parameter/state_dict keys.
        if use_traj:
            # Lazy import so the all-OFF path has no dependency on traj_data.
            from src.traj_data.perceiver import PerceiverResampler

            self.perceiver = PerceiverResampler(
                input_dim=traj_dim,
                hidden_size=self.hidden_size,
                num_latents=perceiver_latents,
                depth=perceiver_depth,
                num_heads=perceiver_heads,
                max_seq_len=perceiver_max_seq_len,
            )
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
        **kwargs,
    ):
        # Fast-path: numerically identical to parent when no fusion feature is active.
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
            traj_lat = self.perceiver(traj_embeds.to(param_dtype), key_padding_mask=traj_mask)
            seq_parts.append(traj_lat)
            pad_parts.append(torch.zeros(B, traj_lat.shape[1], device=device, dtype=torch.bool))
            stream_ids += [3] * traj_lat.shape[1]

        seq = torch.cat(seq_parts, dim=1)
        src_key_padding_mask = torch.cat(pad_parts, dim=1)

        if self.stream_type_emb:
            ids = torch.tensor(stream_ids, device=device)
            seq = seq + self.stream_type_embedding(ids).unsqueeze(0)

        encoded = self.context_encoder(seq, src_key_padding_mask=src_key_padding_mask)
        layer_ctx = self.context_dropout(encoded[:, : self.num_layers, :])
        return self._emit_lora(layer_ctx, B)
