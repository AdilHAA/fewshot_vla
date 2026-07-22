"""TrajHyperLoRASmolVLAPolicy — trajectory-conditioned Hyper-LoRA (leave-one-out).

A backward-compatible SUBCLASS of `HyperLoRASmolVLAPolicy`. The parent files are not
touched. With every new knob at its default (`hn_use_traj_clip=False`) this policy is
structurally identical to `hyper_lora_smolvla` (the `FusionHyperNetwork` builds no extra
params and `_inject_lora` passes no `traj_embeds`).

When `hn_use_traj_clip=True` this policy conditions the hypernetwork on task demos read
from the offline `TrajCache` (`hn_xpair_cache_path`); `traj_dim` comes from the cache
header, so no encoder model is loaded at construction or during training:
  * TRAIN: `_inject_lora` reads the context demo chosen by the p_self selector
    (`hn_p_self`: prob of the imitated episode itself; 0.0 = one other same-task
    demo, the off-diagonal of the within-task cartesian product);
  * EVAL: the base task is resolved (batch `task_index`, else the decoded instruction
    matched against the cache `task_texts`) and its K cached demos are read
    deterministically, then expanded across the vectorized env batch.

The generated adapter is cached per episode via `HN_LORA_CACHE=episode` (parent knob), so
the demo read + hypernetwork run happen once per rollout episode. The verified pure logic
lives in `src/traj_data/` (selectors, fusion forward, encoder, cache); this file is the
lerobot policy glue.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from src.hyper_lora.dynamic_lora import DynamicLoRALinear
from src.hyper_lora.modeling_hyper_lora_smolvla import HyperLoRASmolVLAPolicy
from src.traj_data.xpair_select import (
    pack_conditioning,
    select_eval_conditioning,
    select_train_conditioning,
)

from .configuration_traj_hyper_lora_smolvla import TrajHyperLoRASmolVLAConfig
from .fusion_hypernetwork import FusionHyperNetwork

logger = logging.getLogger(__name__)


class TrajHyperLoRASmolVLAPolicy(HyperLoRASmolVLAPolicy):
    config_class = TrajHyperLoRASmolVLAConfig
    name = "traj_hyper_lora_smolvla"

    def __init__(self, config: TrajHyperLoRASmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._traj_cache = None
        traj_dim = 0
        if config.hn_use_traj_clip:
            if not config.hn_xpair_cache_path:
                raise ValueError("hn_use_traj_clip=True requires hn_xpair_cache_path")
            from src.traj_data.encoder import encoder_format
            from src.traj_data.traj_cache import TrajCache

            self._traj_cache = TrajCache(config.hn_xpair_cache_path)
            self._traj_cache.assert_header_matches(
                encoder_id=config.hn_traj_encoder,
                format=encoder_format(config.hn_traj_encoder,
                                      getattr(config, "hn_vjepa_grid", 2)))
            traj_dim = int(self._traj_cache.header["d_enc"])

        tm = self.hypernet.target_modules
        dino_dim = (int(self.dino.config.hidden_size)
                    if getattr(self, "dino", None) is not None else 0)
        self.hypernet = FusionHyperNetwork(
            text_embed_dim=self._vlm_text_hidden_size(),
            hidden_size=config.hn_hidden_size,
            num_layers=len(self._vlm_text_model().layers),
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=tm,
            dropout=config.hn_dropout,
            encoder_type=config.hn_encoder_type,
            tf_num_blocks=config.hn_tf_num_blocks,
            tf_num_heads=config.hn_tf_num_heads,
            use_vlm_vision=config.hn_use_vlm_vision,
            vlm_vision_dim=self._vlm_text_hidden_size(),
            use_dino=config.hn_use_dino,
            dino_dim=dino_dim,
            zero_init_up=config.hn_zero_init_up,
            use_traj=config.hn_use_traj_clip,
            traj_dim=traj_dim,
            stream_type_emb=config.hn_stream_type_emb,
            per_stream_null=config.hn_per_stream_null,
            readout=config.hn_readout,
        )
        self._freeze_base()
        self._traj_gen = torch.Generator().manual_seed(int(config.hn_seed))

    # --- backward-compatible site patching (unchanged from Stage 1) -------------------
    def _patch_mlp_layers(
        self, config: TrajHyperLoRASmolVLAConfig
    ) -> Dict[str, Tuple[int, int]]:
        new_sites = bool(getattr(config, "hn_inject_vlm_kv", False)) or bool(
            getattr(config, "hn_inject_expert_q", False)
        )
        text_model = self._vlm_text_model()
        target_modules: Dict[str, Tuple[int, int]] = {}

        def _wrap(parent_mod: nn.Module, mod: str, key: str, layer_idx: int) -> None:
            base = getattr(parent_mod, mod)
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"{key} (layer {layer_idx}): expected nn.Linear, got {type(base).__name__}"
                )
            wrapper = DynamicLoRALinear(
                base_layer=base, lora_rank=config.lora_rank, lora_alpha=config.lora_alpha
            )
            setattr(parent_mod, mod, wrapper)
            self._patched.setdefault(key, {})[layer_idx] = wrapper
            target_modules[key] = (base.in_features, base.out_features)

        for i, layer in enumerate(text_model.layers):
            if getattr(config, "hn_inject_vlm_mlp", True):
                for mod in config.hn_target_module_names:
                    _wrap(layer.mlp, mod, f"mlp__{mod}" if new_sites else mod, i)
            if getattr(config, "hn_inject_vlm_kv", False):
                for mod in ("k_proj", "v_proj"):
                    _wrap(layer.self_attn, mod, f"attn__{mod}", i)
        return target_modules

    # --- trajectory-conditioned LoRA injection ---------------------------------------
    def _inject_lora(self, batch: Dict[str, Tensor]) -> None:
        cache_episode = (not self.training) and self._lora_cache_mode() == "episode"
        if cache_episode and self._lora_cache is not None:
            self._set_lora_weights(self._lora_cache)
            return

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.set_grad_enabled(self.training):
            text_embeds = self._embed_language(lang_tokens)

        vlm_vision_embeds = None
        dino_embeds = None
        with torch.no_grad():
            if self.config.hn_use_vlm_vision:
                vlm_vision_embeds = self._vlm_vision_features(batch)
            if self.config.hn_use_dino:
                dino_embeds = self._dino_features(batch)

        traj_embeds = traj_mask = traj_marks = None
        if getattr(self.config, "hn_use_traj_clip", False):
            with torch.no_grad():
                traj_embeds, traj_mask, traj_marks = self._build_traj_conditioning(batch)

        weights = self.hypernet(
            text_embeds, lang_masks, vlm_vision_embeds, dino_embeds,
            traj_embeds=traj_embeds, traj_mask=traj_mask, traj_marks=traj_marks,
        )
        if os.environ.get("HN_LOG_LORA"):
            self._log_lora_drift(weights)
        self._set_lora_weights(weights)
        if cache_episode:
            self._lora_cache = weights

    def _hypernet_device(self) -> torch.device:
        return next(self.hypernet.parameters()).device

    def _build_traj_conditioning(self, batch: Dict[str, Tensor]):
        dev = self._hypernet_device()
        if self.training:
            ep, t = batch["episode_index"], batch["task_index"]
            ep = ep.tolist() if hasattr(ep, "tolist") else list(ep)
            t = t.tolist() if hasattr(t, "tolist") else list(t)
            traj, mask, marks = select_train_conditioning(
                self._traj_cache, ep, t, self.config.hn_p_self,
                self.config.hn_context_k, self._traj_gen)
            return traj.to(dev), mask.to(dev), marks.to(dev)
        # EVAL: every env in a rollout batch runs the same suite task -> one lookup,
        # deterministic demo pick, expanded across the env batch. Context size equals
        # hn_context_k for every arm — the train/eval format always matches.
        task_index = self._resolve_eval_task(batch)
        demos = select_eval_conditioning(
            self._traj_cache, task_index, self.config.hn_context_k, self.config.hn_seed)
        bsz = batch[OBS_LANGUAGE_TOKENS].shape[0]
        traj, mask, marks = pack_conditioning([demos] * bsz)
        logger.warning("[TRAJ] task=%d demos=%d tokens=%d",
                       task_index, len(demos), traj.shape[1])
        return traj.to(dev), mask.to(dev), marks.to(dev)

    def _resolve_eval_task(self, batch: Dict[str, Tensor]) -> int:
        """Base-task lookup: batch task_index when present, else the instruction
        (raw string key, or decoded tokens) matched against the cache's task_texts."""
        v = batch.get("task_index") if isinstance(batch, dict) else None
        if v is not None:
            return int(v.flatten()[0]) if hasattr(v, "flatten") else int(v)
        # Raw instruction string — the env usually exposes it under "task".
        text = ""
        for key in ("task", "instruction", "language_instruction", "prompt"):
            t = batch.get(key) if isinstance(batch, dict) else None
            if isinstance(t, (list, tuple)) and t:
                t = t[0]
            if isinstance(t, str) and t:
                text = t
                break
        # Fall back to decoding the tokenized instruction with a lazily-built tokenizer.
        if not text and isinstance(batch, dict) and OBS_LANGUAGE_TOKENS in batch:
            text = self._decode_instruction(batch[OBS_LANGUAGE_TOKENS][0])
        ti = self._traj_cache.resolve_task(text)
        if ti is None:
            # Novel-instruction suites (e.g. the _task axis) legitimately miss the
            # cutoff — condition on the nearest known task instead of crashing.
            ti = self._traj_cache.nearest_task(text)
            if ti is not None:
                logger.warning("[TRAJ] instruction %r matched no cached task; "
                               "falling back to nearest task %d", text, ti)
        if ti is None:
            keys = sorted(batch.keys()) if isinstance(batch, dict) else type(batch).__name__
            raise RuntimeError(f"cannot resolve eval task; instruction={text!r}, "
                               f"batch keys={keys}; pass task_index or extend task_texts")
        return ti

    def _decode_instruction(self, token_ids) -> str:
        """Decode OBS_LANGUAGE_TOKENS via a tokenizer loaded from the VLM (cached).
        The policy holds no tokenizer of its own, so build one on first eval use."""
        if getattr(self, "_lang_tok", None) is None:
            from transformers import AutoTokenizer

            vlm_id = (getattr(self.config, "vlm_model_name", None)
                      or getattr(self.config, "load_vlm_weights_from", None)
                      or "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            try:
                self._lang_tok = AutoTokenizer.from_pretrained(vlm_id)
            except Exception as e:               # decode is best-effort; task key usually wins
                logger.warning("could not load tokenizer %s for eval task decode: %s", vlm_id, e)
                self._lang_tok = False
        if not self._lang_tok:
            return ""
        return self._lang_tok.decode(token_ids, skip_special_tokens=True).strip()
