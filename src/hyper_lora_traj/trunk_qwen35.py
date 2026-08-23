"""TrunkHyperNetwork — a pretrained trunk in place of the scratch hypernetwork.

The brief (direction 7): use a ready VLM to process the trajectory video AND the
instruction at once, add 32 trainable layer tokens, and train those. Qwen3.5 is
natively multimodal, and its vision tower attends PER FRAME (cu_seqlens segment each
frame's patches — modeling_qwen3_5.py builds them with repeat_interleave(H*W, T)),
so the visual tokens are a pure function of the frames. They are therefore computed
ONCE into an offline cache (`build_qwen35_video_cache.py`) and reused: no ViT and no
video decode at train time. The vision merger already outputs LLM-dim tokens, so the
cached vectors go into the trunk unprojected.

What runs per step here is only the frozen text stack (Qwen3_5TextModel, ~752M) over

    [ left-pad | cached video tokens | text embeddings | 32 layer tokens (END) ]

Attention is causal, so the layer tokens sit at the very end where they see
everything. Padding is LEFT so their tail positions stay aligned batch-wide;
verified: perturbing left-pad slots changes the readout by exactly 0.0. Readout =
last_hidden_state[:, -32:] -> trunk_proj(trunk_dim->128) -> the PARENT's
head_down/head_up -> the same LoRA weights, so the only variable vs the scratch
arms is the trunk itself.

Gradient reachability (measured on the real checkpoint): the 32 tokens train through
a fully frozen trunk ONLY when the trunk forward runs under autograd — hence
gradient checkpointing + train() mode here; without it, activations run ~5.8
MB/token (~93 GB at B=32, L=500) and do not fit an 80 GB H100.

NOTE the GatedDeltaNet layers need `flash-linear-attention` + `causal-conv1d`
installed, or the trunk falls back to a slow fp32 chunked path.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from src.hyper_lora.hypernetwork import HyperNetwork

logger = logging.getLogger(__name__)


class TrunkHyperNetwork(HyperNetwork):
    """Same heads/LoRA contract as the parent; the conditioning trunk is pretrained.

    Instantiated ONLY when hn_trunk_model is set — every existing arm keeps building
    HyperNetwork/FusionHyperNetwork, so their byte-compat guarantees are untouched.
    The parent's context encoder/text projections are still constructed (inherited
    __init__) but unused: inert parameters in trunk-arm checkpoints only, which also
    keeps `heads_down/heads_up` and `_emit_lora` reused verbatim.
    """

    def __init__(self, *args, trunk_model: str, trunk_dim: int, trunk_text: bool = True,
                 trunk_grad_ckpt: bool = True, trunk_q_std: float = 0.02,
                 use_traj: bool = False, traj_dim: int = 0,
                 stream_type_emb: bool = False, per_stream_null: bool = False,
                 readout: str = "queries", **kwargs):
        # use_traj/traj_dim/stream_type_emb/per_stream_null/readout are
        # FusionHyperNetwork's knobs, accepted here so the policy can construct either
        # class with one call shape; for the trunk they are all implied and ignored.
        super().__init__(*args, **kwargs)
        self.trunk_model = trunk_model
        self.trunk_text = trunk_text
        self.trunk_grad_ckpt = trunk_grad_ckpt
        self.trunk_dim = int(trunk_dim)
        # Learned pieces (everything else is frozen):
        #   32 layer tokens, in the trunk's own space, ADDED at the sequence end;
        #   one projection trunk_dim -> 128 feeding the unchanged LoRA heads.
        self.layer_queries = nn.Parameter(torch.randn(self.num_layers, self.trunk_dim)
                                          * trunk_q_std)
        self.trunk_proj = nn.Linear(self.trunk_dim, self.hidden_size)
        self._trunk = None            # lazy: transformers + weights, never at CI

    # --- lazy trunk ---------------------------------------------------------------
    def _ensure_trunk(self, device):
        if self._trunk is not None:
            return
        from transformers import AutoTokenizer, Qwen3_5TextModel

        trunk = Qwen3_5TextModel.from_pretrained(self.trunk_model, dtype=torch.bfloat16)
        if int(getattr(trunk.config, "hidden_size", self.trunk_dim)) != self.trunk_dim:
            raise ValueError(
                f"trunk {self.trunk_model} hidden={trunk.config.hidden_size} != "
                f"trunk_dim={self.trunk_dim} (the video cache was built for another "
                "trunk size — rebuild the cache or change hn_trunk_model)")
        for p in trunk.parameters():
            p.requires_grad_(False)
        if self.trunk_grad_ckpt:
            trunk.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        trunk.train()                 # HF checkpointing only engages in train mode
        self._trunk = trunk.to(device)
        self._trunk_tokenizer = AutoTokenizer.from_pretrained(self.trunk_model)
        logger.info("[TRUNK] %s: %d layers, text=%s, grad_ckpt=%s",
                    self.trunk_model, trunk.config.num_hidden_layers,
                    self.trunk_text, self.trunk_grad_ckpt)

    # --- forward -------------------------------------------------------------------
    def forward_trunk(self, video_tokens: Tensor, video_pad: Tensor, texts: list[str]):
        """video_tokens (B,L,d_enc) — cached Qwen video tokens (LLM-dim already);
        video_pad (B,L) True==pad; texts — one instruction per sample.

        Returns {mod: {layer: (W_down, W_up)}}, the same contract as every
        hypernetwork, so the policy's LoRA injection path is unchanged."""
        B, device = video_tokens.shape[0], video_tokens.device
        self._ensure_trunk(device)
        dtype = next(self._trunk.parameters()).dtype

        # Per-row segments, LEFT-padded to ONE batch-wide length:
        #   [ pad | video | text | 32 layer queries ]  (queries always at the tail)
        rows_segs = []
        for b in range(B):
            keep = ~video_pad[b].bool()
            segs = [video_tokens[b][keep].to(dtype)]
            if self.trunk_text and texts[b]:
                ids = self._trunk_tokenizer(texts[b], return_tensors="pt").input_ids.to(device)
                segs.append(self._trunk.get_input_embeddings()(ids)[0].to(dtype))
            segs.append(self.layer_queries.to(dtype))
            rows_segs.append(segs)
        Lmax = max(sum(s.shape[0] for s in segs) for segs in rows_segs)
        # Round the padded length UP to a multiple of 256: sequence lengths vary
        # batch to batch (880..6200), and every distinct length triggers a fresh
        # Triton JIT compile inside the trunk (~seconds each, observed as ~27s
        # stalls every few steps). Bucketing collapses the shape space to ~25
        # values, so compiles stop after a short warmup. The extra pad slots are
        # masked and provably cannot reach the readout.
        Lmax = ((Lmax + 255) // 256) * 256
        rows, masks = [], []
        for segs in rows_segs:
            L = sum(s.shape[0] for s in segs)
            row = torch.zeros(Lmax, self.trunk_dim, dtype=dtype, device=device)
            m = torch.zeros(Lmax, dtype=torch.bool, device=device)
            off = Lmax                               # fill from the tail: left-pad
            for s in reversed(segs):
                off -= s.shape[0]
                row[off:off + s.shape[0]] = s
                m[off:off + s.shape[0]] = True
            rows.append(row)
            masks.append(m)
        seq = torch.stack(rows)
        attn = torch.stack(masks)                    # 1=keep, 0=left-pad

        # At train the graph MUST pass through the frozen trunk (that is how the 32
        # layer tokens learn). At eval nothing needs gradients, but layer_queries is
        # a leaf parameter, so an unguarded call would still build the graph and pin
        # ~17 GB of trunk activations per episode — free memory and speed for nothing.
        with (torch.enable_grad() if self.training else torch.no_grad()):
            out = self._trunk(inputs_embeds=seq, attention_mask=attn, use_cache=False)
        ctx = out.last_hidden_state[:, -self.num_layers:, :]
        ctx = self.trunk_proj(ctx.to(self.trunk_proj.weight.dtype))
        return self._emit_lora(self.context_dropout(ctx), B)
