import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class HyperNetwork(nn.Module):
    """Generates LoRA (W_down, W_up) per (target_module, layer_idx) from the
    instruction, optionally also conditioned on visual context.

    Conditioning sources (any subset, concatenated as extra context tokens):
      * text         — VLM language-token embeddings           (always on)
      * vlm_vision    — the frozen VLM's own image tokens        (`use_vlm_vision`)
      * dino          — an external frozen DINOv2's patch tokens (`use_dino`)

    Encoder is `tf` (layer-position tokens prepended to the context sequence;
    spatial vision tokens are kept so positional cues survive) or `mlp`
    (mean-pooled context + layer embedding). Output heads are shared across layers
    but distinct per target_module (in/out dims differ).
    """

    def __init__(
        self,
        text_embed_dim: int = 960,
        layer_embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 16,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        target_modules: Optional[Dict[str, Tuple[int, int]]] = None,
        dropout: float = 0.1,
        encoder_type: str = "tf",
        tf_num_blocks: int = 2,
        tf_num_heads: int = 4,
        # vision-conditioning; 0-dim or False => disabled
        use_vlm_vision: bool = False,
        vlm_vision_dim: int = 0,
        use_dino: bool = False,
        dino_dim: int = 0,
        zero_init_up: bool = True,
    ):
        super().__init__()
        if target_modules is None:
            target_modules = {"gate_proj": (960, 2560)}

        self.num_layers = num_layers
        self.target_modules = target_modules
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size
        self.use_vlm_vision = use_vlm_vision
        self.use_dino = use_dino

        # Each conditioning source gets its own projection into `hidden_size`.
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)
        if use_vlm_vision:
            self.vlm_vision_proj = nn.Linear(vlm_vision_dim, hidden_size)
        if use_dino:
            self.dino_proj = nn.Linear(dino_dim, hidden_size)
        # number of extra (vision) context streams, for the mlp input width
        n_vision_streams = int(use_vlm_vision) + int(use_dino)

        if encoder_type == "mlp":
            self.layer_embedding = nn.Embedding(num_layers, layer_embed_dim)
            ctx_in = hidden_size + layer_embed_dim + n_vision_streams * hidden_size
            self.context_encoder = nn.Sequential(
                nn.Linear(ctx_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(dropout),
            )
        elif encoder_type == "tf":
            self.layer_embedding = nn.Embedding(num_layers, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=tf_num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=tf_num_blocks
            )
            self.context_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.heads_up = nn.ModuleDict()
        self.heads_down = nn.ModuleDict()
        for mod, (in_f, out_f) in target_modules.items():
            self.heads_down[mod] = nn.Linear(hidden_size, self.lora_rank * in_f)
            self.heads_up[mod] = nn.Linear(hidden_size, out_f * self.lora_rank)
            if zero_init_up:
                # W_up ≡ 0 at init => ΔW = W_up·W_down = 0, so the policy starts
                # exactly equal to the frozen base (LoRA's B=0 convention).
                # Without this the random ΔW (scaled by alpha/rank) corrupts the
                # ~90%-SR base at step 0 and training must first undo the damage.
                nn.init.zeros_(self.heads_up[mod].weight)
                nn.init.zeros_(self.heads_up[mod].bias)

    def forward(
        self,
        text_token_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vlm_vision_embeds: Optional[torch.Tensor] = None,
        dino_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        text_token_embeds: (B, T, text_embed_dim)
        attention_mask:    (B, T), 1 for real tokens, 0 for padding (or None)
        vlm_vision_embeds: (B, Nv, vlm_vision_dim) or None  (used iff use_vlm_vision)
        dino_embeds:       (B, Nd, dino_dim)       or None  (used iff use_dino)
        """
        if self.encoder_type == "mlp":
            return self._forward_mlp(
                text_token_embeds, attention_mask, vlm_vision_embeds, dino_embeds
            )
        return self._forward_tf(
            text_token_embeds, attention_mask, vlm_vision_embeds, dino_embeds
        )

    # --- helpers -------------------------------------------------------------
    def _proj_vision(self, vlm_vision_embeds, dino_embeds, dtype):
        """Project whichever vision streams are enabled into hidden_size.
        Returns a list of (B, N_i, hidden) tensors (possibly empty)."""
        streams = []
        if self.use_vlm_vision and vlm_vision_embeds is not None:
            streams.append(self.vlm_vision_proj(vlm_vision_embeds.to(dtype)))
        if self.use_dino and dino_embeds is not None:
            streams.append(self.dino_proj(dino_embeds.to(dtype)))
        return streams

    def _emit_lora(self, layer_ctx, B):
        """layer_ctx: (B, num_layers, hidden) -> per-(mod, layer) LoRA pair."""
        lora_weights = {mod: {} for mod in self.target_modules}
        for layer_idx in range(self.num_layers):
            ctx = layer_ctx[:, layer_idx, :]
            for mod, (in_f, out_f) in self.target_modules.items():
                w_d = self.heads_down[mod](ctx).view(B, self.lora_rank, in_f)
                w_u = self.heads_up[mod](ctx).view(B, out_f, self.lora_rank)
                lora_weights[mod][layer_idx] = (w_d, w_u)
        return lora_weights

    # --- encoders ------------------------------------------------------------
    def _forward_mlp(self, text_token_embeds, attention_mask, vlm_vision_embeds, dino_embeds):
        param_dtype = self.text_proj.weight.dtype
        text = self.text_proj(text_token_embeds.to(param_dtype))
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(text.dtype)
            text_pooled = (text * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            text_pooled = text.mean(dim=1)

        # Mean-pool each vision stream into a single context vector and append.
        pooled = [text_pooled]
        for v in self._proj_vision(vlm_vision_embeds, dino_embeds, param_dtype):
            pooled.append(v.mean(dim=1))
        ctx_vec = torch.cat(pooled, dim=-1)

        B = text_pooled.size(0)
        layer_ctx_list = []
        for layer_idx in range(self.num_layers):
            l_idx = torch.tensor([layer_idx], device=text_pooled.device).expand(B)
            l_emb = self.layer_embedding(l_idx)
            ctx = self.context_encoder(torch.cat([ctx_vec, l_emb], dim=-1))
            layer_ctx_list.append(ctx)
        layer_ctx = torch.stack(layer_ctx_list, dim=1)  # (B, num_layers, hidden)
        return self._emit_lora(layer_ctx, B)

    def _forward_tf(self, text_token_embeds, attention_mask, vlm_vision_embeds, dino_embeds):
        B = text_token_embeds.shape[0]
        device = text_token_embeds.device
        param_dtype = self.text_proj.weight.dtype

        text_h = self.text_proj(text_token_embeds.to(param_dtype))

        layer_idx_t = torch.arange(self.num_layers, device=device)
        layer_h = self.layer_embedding(layer_idx_t).unsqueeze(0).expand(B, -1, -1)

        # Context sequence: [layer tokens | text tokens | vision tokens...].
        # Layer tokens are the queries we read back; everything else is context.
        seq_parts = [layer_h, text_h]
        # padding mask: True == ignore. Layer tokens are always valid.
        pad_parts = [torch.zeros(B, self.num_layers, device=device, dtype=torch.bool)]
        if attention_mask is not None:
            pad_parts.append(~attention_mask.bool())
        else:
            pad_parts.append(torch.zeros(B, text_h.shape[1], device=device, dtype=torch.bool))

        for v in self._proj_vision(vlm_vision_embeds, dino_embeds, param_dtype):
            seq_parts.append(v)
            pad_parts.append(torch.zeros(B, v.shape[1], device=device, dtype=torch.bool))

        seq = torch.cat(seq_parts, dim=1)
        src_key_padding_mask = torch.cat(pad_parts, dim=1)

        encoded = self.context_encoder(seq, src_key_padding_mask=src_key_padding_mask)
        layer_ctx = self.context_dropout(encoded[:, : self.num_layers, :])
        return self._emit_lora(layer_ctx, B)


class StaticLoRABank(nn.Module):
    """MT-LoRA baseline: one ordinary learnable LoRA pair per (target_module,
    layer), shared across all tasks — no conditioning of any kind.

    This is the key control for the hypernetwork: it trains on the same data,
    through the same DynamicLoRALinear injection sites, with the same rank/alpha
    and the same optimizer budget. Any HN gain over this baseline is attributable
    to input-conditioning rather than to the extra adaptation training itself.

    Mirrors HyperNetwork's output contract: forward(batch_size) returns
    {mod: {layer_idx: (W_down (B, r, in), W_up (B, out, r))}} with the static
    weights expanded (stride-0, no copy) over the batch.

    Init follows standard LoRA: W_down ~ Kaiming-uniform, W_up = 0.
    """

    def __init__(
        self,
        num_layers: int,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        target_modules: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()
        if target_modules is None:
            target_modules = {"gate_proj": (960, 2560)}
        self.num_layers = num_layers
        self.target_modules = target_modules
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.w_down = nn.ParameterDict()
        self.w_up = nn.ParameterDict()
        for mod, (in_f, out_f) in target_modules.items():
            for layer_idx in range(num_layers):
                key = f"{mod}_{layer_idx}"
                w_d = torch.empty(self.lora_rank, in_f)
                nn.init.kaiming_uniform_(w_d, a=5**0.5)
                self.w_down[key] = nn.Parameter(w_d)
                self.w_up[key] = nn.Parameter(torch.zeros(out_f, self.lora_rank))

    def forward(self, batch_size: int) -> Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        weights: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
            mod: {} for mod in self.target_modules
        }
        for mod in self.target_modules:
            for layer_idx in range(self.num_layers):
                key = f"{mod}_{layer_idx}"
                w_d = self.w_down[key].unsqueeze(0).expand(batch_size, -1, -1)
                w_u = self.w_up[key].unsqueeze(0).expand(batch_size, -1, -1)
                weights[mod][layer_idx] = (w_d, w_u)
        return weights
