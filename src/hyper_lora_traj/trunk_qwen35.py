"""TrunkHyperNetwork — a pretrained trunk in place of the scratch hypernetwork.

The brief (direction 7): use a ready VLM to process the trajectory video AND the
instruction at once, add one trainable layer token per LoRA layer, and train those.
Qwen3.5 is natively multimodal, and its vision tower attends PER FRAME (cu_seqlens
segment each frame's patches — modeling_qwen3_5.py builds them with
repeat_interleave(H*W, T)), so the visual tokens are a pure function of the frames.
They are therefore computed ONCE into an offline cache
(`build_qwen35_video_cache.py`) and reused: no ViT and no video decode at train
time. The vision merger already outputs LLM-dim tokens, so the cached vectors go
into the trunk unprojected.

What runs per step here is only the frozen text stack (Qwen3_5TextModel, ~752M).
The sequence it runs on comes in two layouts:

  native (default) — the prompt a Qwen3.5 chat turn would really contain:

    [ left-pad | <|im_start|>user\\n (<t s><|vision_start|> 49 tok <|vision_end|>)*
                 instruction <|im_end|>\\n<|im_start|>assistant\\n | layer tokens ]

    with 3-D M-RoPE positions, so a temporal patch's 7x7 tokens share one time index
    exactly as get_rope_index would place them (see qwen_video_layout.py). The
    checkpoint has only ever seen video in this scaffolding; feeding bare tokens at
    1-D positions is off-distribution for free.

  raw (trunk_native=False) — the original layout, kept so the arms trained before
    the native path stay reproducible:

    [ left-pad | all demo tokens | text embeddings | layer tokens ]

Attention is causal, so the layer tokens sit at the very end where they see
everything. Padding is LEFT so their tail positions stay aligned batch-wide;
verified: perturbing left-pad slots changes the readout by exactly 0.0. Readout =
last_hidden_state[:, -num_layers:] -> trunk_proj(trunk_dim->128) -> the PARENT's
head_down/head_up -> the same LoRA weights, so the only variable vs the scratch
arms is the trunk itself.

Gradient reachability (measured on the real checkpoint): the layer tokens train
through a fully frozen trunk ONLY when the trunk forward runs under autograd — hence
gradient checkpointing + train() mode here; without it, activations run ~5.8
MB/token (~93 GB at B=32, L=500) and do not fit an 80 GB H100.

NOTE the GatedDeltaNet layers need `flash-linear-attention` + `causal-conv1d`
installed, or the trunk falls back to a slow fp32 chunked path.
"""
from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

from src.hyper_lora.hypernetwork import HyperNetwork
from src.hyper_lora_traj.qwen_video_layout import (
    build_prompt_text,
    four_row_positions,
    left_pad_rows,
    mrope_positions,
)

logger = logging.getLogger(__name__)

Clip = list[tuple[Tensor, list[float]]]        # one sample: k demos (tokens, stamps)


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
                 trunk_native: bool = True,
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
        self.trunk_native = trunk_native
        self.trunk_dim = int(trunk_dim)
        # Learned pieces (everything else is frozen):
        #   one layer token per LoRA layer, in the trunk's own space, ADDED at the
        #   sequence end; one projection trunk_dim -> 128 feeding the LoRA heads.
        self.layer_queries = nn.Parameter(torch.randn(self.num_layers, self.trunk_dim)
                                          * trunk_q_std)
        self.trunk_proj = nn.Linear(self.trunk_dim, self.hidden_size)
        self._trunk = None            # lazy: transformers + weights, never at CI
        self._trunk_tokenizer = None
        # Filled from the trunk config by _ensure_trunk (tests stub them directly).
        self._video_token_id = None
        self._vision_start_token_id = None
        self._vision_end_token_id = None
        self._spatial_merge_size = None

    # --- lazy trunk ---------------------------------------------------------------
    def _ensure_trunk(self, device):
        if self._trunk is not None:
            return
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel

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
        # The vision ids live on the TOP-LEVEL Qwen3_5 config, not on the text config
        # the loaded Qwen3_5TextModel carries — hence the separate AutoConfig load.
        cfg = AutoConfig.from_pretrained(self.trunk_model)
        self._video_token_id = int(cfg.video_token_id)
        self._vision_start_token_id = int(cfg.vision_start_token_id)
        self._vision_end_token_id = int(cfg.vision_end_token_id)
        self._spatial_merge_size = int(cfg.vision_config.spatial_merge_size)
        logger.info("[TRUNK] %s: %d layers, text=%s, native=%s, grad_ckpt=%s",
                    self.trunk_model, trunk.config.num_hidden_layers,
                    self.trunk_text, self.trunk_native, self.trunk_grad_ckpt)

    # --- row assembly ----------------------------------------------------------------
    def _raw_row(self, clip: Clip, text: str, device, dtype):
        """Legacy layout: demo tokens concatenated bare, then text, then the queries."""
        segs = [tok.to(device=device, dtype=dtype) for tok, _ in clip]
        if self.trunk_text and text:
            ids = self._trunk_tokenizer(text, return_tensors="pt").input_ids.to(device)
            segs.append(self._trunk.get_input_embeddings()(ids)[0].to(dtype))
        segs.append(self.layer_queries.to(dtype))
        return torch.cat(segs), None

    def _native_row(self, clip: Clip, text: str, device, dtype):
        """Build the real chat prompt, then substitute the cached tokens into it."""
        units = [len(stamps) for _, stamps in clip]
        tpu = clip[0][0].shape[0] // max(units[0], 1)         # 49 tokens per pair
        for (tok, stamps), n in zip(clip, units):
            if n == 0 or tok.shape[0] != tpu * n:
                raise ValueError(f"demo has {tok.shape[0]} tokens for {n} timestamps "
                                 f"(expected {tpu} per timestamp)")

        prompt = build_prompt_text([s for _, s in clip],
                                   text if self.trunk_text else None, tpu)
        ids = self._trunk_tokenizer(prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids[0].to(device)
        video = ids == self._video_token_id
        cached = torch.cat([tok.to(device=device, dtype=dtype) for tok, _ in clip])
        if int(video.sum()) != cached.shape[0]:
            raise ValueError(f"prompt has {int(video.sum())} video slots but the cache "
                             f"supplies {cached.shape[0]} tokens (tokenizer dropped or "
                             "merged <|video_pad|>?)")
        for name, tid in (("start", self._vision_start_token_id),
                          ("end", self._vision_end_token_id)):
            if int((ids == tid).sum()) != sum(units):
                raise ValueError(f"prompt has {int((ids == tid).sum())} vision_{name} "
                                 f"markers for {sum(units)} temporal patches")

        emb = self._trunk.get_input_embeddings()(ids).to(dtype)
        emb = emb.masked_scatter(video.unsqueeze(-1).expand_as(emb), cached)
        emb = torch.cat([emb, self.layer_queries.to(dtype)])

        # mm_token_type_ids: 2 on the video slots, 0 everywhere else (the timestamps
        # and the vision markers are text) — the layer queries are text too.
        types = torch.zeros(ids.shape[0], dtype=torch.long)
        types[video.cpu()] = 2
        types = torch.cat([types, torch.zeros(self.num_layers, dtype=torch.long)])
        side = math.isqrt(tpu) * self._spatial_merge_size
        grids = [(n, side, side) for n in units]              # patch units per demo
        pos3 = mrope_positions(types, grids, self._spatial_merge_size)
        return emb, four_row_positions(pos3).to(device)

    # --- forward -------------------------------------------------------------------
    def forward_trunk(self, clips: list[Clip], texts: list[str]):
        """clips — per sample, a list of k demos (cached tokens (L,d_enc) fp16, one
        timestamp per 49-token temporal patch); texts — one instruction per sample.

        Returns {mod: {layer: (W_down, W_up)}}, the same contract as every
        hypernetwork, so the policy's LoRA injection path is unchanged."""
        B = len(clips)
        device = self.trunk_proj.weight.device
        self._ensure_trunk(device)
        dtype = next(self._trunk.parameters()).dtype

        build = self._native_row if self.trunk_native else self._raw_row
        seq, attn, pos = left_pad_rows([build(clips[b], texts[b], device, dtype)
                                        for b in range(B)])

        # At train the graph MUST pass through the frozen trunk (that is how the layer
        # tokens learn). At eval nothing needs gradients, but layer_queries is a leaf
        # parameter, so an unguarded call would still build the graph and pin ~17 GB
        # of trunk activations per episode — free memory and speed for nothing.
        with (torch.enable_grad() if self.training else torch.no_grad()):
            out = self._trunk(inputs_embeds=seq, attention_mask=attn,
                              position_ids=pos, use_cache=False)
        ctx = out.last_hidden_state[:, -self.num_layers:, :]
        ctx = self.trunk_proj(ctx.to(self.trunk_proj.weight.dtype))
        return self._emit_lora(self.context_dropout(ctx), B)
