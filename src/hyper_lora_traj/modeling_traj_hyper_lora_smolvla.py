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
  * EVAL: the task key (the LIBERO bddl stem `eval_hyper_lora.py` forwards as
    `batch['subtask']`) selects that task's K cached demos deterministically, then
    they are expanded across the vectorized env batch.

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
from src.traj_data.stride import pair_timestamps
from src.traj_data.xpair_select import pack_rows, select_eval_rows, select_train_rows

from .configuration_traj_hyper_lora_smolvla import TrajHyperLoRASmolVLAConfig
from .eval_keys import resolve_eval_key
from .fusion_hypernetwork import FusionHyperNetwork

logger = logging.getLogger(__name__)


class TrajHyperLoRASmolVLAPolicy(HyperLoRASmolVLAPolicy):
    config_class = TrajHyperLoRASmolVLAConfig
    name = "traj_hyper_lora_smolvla"

    def __init__(self, config: TrajHyperLoRASmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._traj_cache = None
        traj_dim = 0
        trunk_model = getattr(config, "hn_trunk_model", "")
        if config.hn_use_traj_clip:
            if not config.hn_xpair_cache_path:
                raise ValueError("hn_use_traj_clip=True requires hn_xpair_cache_path")
            from src.traj_data.encoder import encoder_format
            from src.traj_data.traj_cache import TrajCache

            self._traj_cache = TrajCache(config.hn_xpair_cache_path)
            # The format tag carries encoder, grid AND temporal scope, so a
            # 1-token minimal-clip cache can no longer be silently accepted by a
            # full-clip config (and vice versa). encoder_model/chunk are opt-in
            # pins: they are invisible in the tokens, so leaving them unset keeps
            # every previously trained checkpoint loadable.
            if trunk_model:
                # A trunk arm reads a qwen35vl VIDEO-token cache; the dino/vjepa tag
                # check below would reject it (encoder_id 'dino', format 'cls') before
                # the trunk branch ever ran — this check replaces it.
                expected = {
                    "encoder_id": "qwen35vl",
                    "format": f"qwen35vl_every{int(config.hn_trunk_stride)}",
                    "stride": int(config.hn_trunk_stride),
                    "encoder_model": trunk_model,
                }
            else:
                expected = {
                    "encoder_id": config.hn_traj_encoder,
                    "format": encoder_format(
                        config.hn_traj_encoder,
                        getattr(config, "hn_vjepa_grid", 2),
                        time_select=getattr(config, "hn_traj_time_select", "all"),
                        n_frames=getattr(config, "hn_traj_n_frames", 0),
                        fill=getattr(config, "hn_traj_fill", ""),
                        dino_grid=getattr(config, "hn_dino_grid", 0),
                        include_cls=getattr(config, "hn_dino_include_cls", True),
                        n_reg=getattr(config, "hn_dino_n_reg", 0)),
                }
                if getattr(config, "hn_traj_encoder_model", ""):
                    expected["encoder_model"] = config.hn_traj_encoder_model
                if int(getattr(config, "hn_traj_chunk", 0)):
                    expected["chunk"] = int(config.hn_traj_chunk)
            self._traj_cache.assert_header_matches(**expected)
            traj_dim = int(self._traj_cache.header["d_enc"])

        tm = self.hypernet.target_modules
        dino_dim = (int(self.dino.config.hidden_size)
                    if getattr(self, "dino", None) is not None else 0)
        # DDP: every rank must draw a DIFFERENT context for the same episode,
        # otherwise the ranks of a step condition on one shared demo draw.
        ctx_seed = int(config.hn_seed) + int(os.environ.get("RANK", 0))
        if trunk_model:
            from .trunk_qwen35 import TrunkHyperNetwork
            self.hypernet = TrunkHyperNetwork(
                text_embed_dim=self._vlm_text_hidden_size(),
                hidden_size=config.hn_hidden_size,
                num_layers=len(self._lora_site_layers(config)),
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=tm,
                dropout=config.hn_dropout,
                encoder_type=config.hn_encoder_type,
                tf_num_blocks=config.hn_tf_num_blocks,
                tf_num_heads=config.hn_tf_num_heads,
                use_vlm_vision=False, vlm_vision_dim=self._vlm_text_hidden_size(),
                use_dino=False, dino_dim=0,
                zero_init_up=config.hn_zero_init_up,
                use_traj=True,
                traj_dim=int(self._traj_cache.header["d_enc"]),
                stream_type_emb=False, per_stream_null=False, readout="queries",
                trunk_model=trunk_model,
                trunk_dim=int(self._traj_cache.header["d_enc"]),
                trunk_text=getattr(config, "hn_trunk_text", True),
                trunk_grad_ckpt=getattr(config, "hn_trunk_grad_ckpt", True),
                trunk_native=getattr(config, "hn_trunk_native", True),
            )
            self._freeze_base()
            self._traj_gen = torch.Generator().manual_seed(ctx_seed)
            return
        self.hypernet = FusionHyperNetwork(
            text_embed_dim=self._vlm_text_hidden_size(),
            hidden_size=config.hn_hidden_size,
            num_layers=len(self._lora_site_layers(config)),
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
            traj_pos_emb=getattr(config, "hn_traj_pos_emb", "none"),
            traj_max_pos=getattr(config, "hn_traj_max_pos", 512),
            traj_sub_emb=getattr(config, "hn_traj_sub_emb", False),
            traj_tokens_per_unit=(int(self._traj_cache.header.get("tokens_per_unit", 1))
                                  if self._traj_cache is not None else 1),
            traj_pos_std=getattr(config, "hn_traj_pos_std", 0.02),
        )
        self._freeze_base()
        self._traj_gen = torch.Generator().manual_seed(ctx_seed)

    # --- backward-compatible site patching (unchanged from Stage 1) -------------------
    def _patch_mlp_layers(
        self, config: TrajHyperLoRASmolVLAConfig
    ) -> Dict[str, Tuple[int, int]]:
        if getattr(config, "hn_lora_target", "vlm_mlp") != "vlm_mlp":
            # Expert-site LoRA: the parent branch handles it; the VLM-site flags
            # below are exclusive with it (config __post_init__ enforces that).
            return super()._patch_mlp_layers(config)
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

        # Same init-frame pairing as the parent: with a frame bank configured, the
        # TRAIN conditioning frame is a t=0 frame from the bank (hn_p_self: own
        # episode vs another of the task), not the current step's frame. Without
        # this the VLM stream would see episode E at a random t on train and the
        # rollout's t=0 at eval (HN_LORA_CACHE=episode) — a train/eval mismatch that
        # has nothing to do with what the traj arm is meant to measure.
        cond_batch = batch
        if self.training and getattr(self.config, "hn_frame_bank_path", None):
            cond_batch = self._bank_batch(batch)

        vlm_vision_embeds = None
        dino_embeds = None
        with torch.no_grad():
            if self.config.hn_use_vlm_vision:
                vlm_vision_embeds = self._vlm_vision_features(cond_batch)
            if self.config.hn_use_dino:
                dino_embeds = self._dino_features(cond_batch)

        traj_embeds = traj_mask = traj_marks = traj_pos = traj_sub = None
        trunk = bool(getattr(self.config, "hn_trunk_model", ""))
        rows = keys = None
        if getattr(self.config, "hn_use_traj_clip", False):
            rows, keys = self._context_rows(batch)
            if not trunk:
                dev = self._hypernet_device()
                with torch.no_grad():
                    (traj_embeds, traj_mask, traj_marks, traj_pos, traj_sub) = (
                        x.to(dev) for x in pack_rows(self._traj_cache, rows))

        if trunk:
            # The trunk eats cached VIDEO tokens + the raw instruction string; the
            # SmolVLA text embeds / marks / pos built above are unused here.
            weights = self.hypernet.forward_trunk(self._trunk_clips(rows),
                                                  self._trunk_texts(keys))
        else:
            weights = self.hypernet(
                text_embeds, lang_masks, vlm_vision_embeds, dino_embeds,
                traj_embeds=traj_embeds, traj_mask=traj_mask, traj_marks=traj_marks,
                traj_pos=traj_pos, traj_sub=traj_sub,
            )
        if os.environ.get("HN_LOG_LORA"):
            self._log_lora_drift(weights)
        self._set_lora_weights(weights)
        if cache_episode:
            self._lora_cache = weights

    def _hypernet_device(self) -> torch.device:
        return next(self.hypernet.parameters()).device

    def _context_rows(self, batch: Dict[str, Tensor]) -> tuple[list, list]:
        """Cache rows + task key per batch sample — one selection for both arms.

        TRAIN: the episode's own key (task_index is not a task identity on the merged
        dataset). EVAL: every env of a rollout batch runs the same suite task, so one
        key lookup + one deterministic demo pick, expanded across the env batch;
        context size is hn_context_k on both sides, so the formats always match."""
        if self.training:
            eps = batch["episode_index"]
            eps = [int(e) for e in (eps.tolist() if hasattr(eps, "tolist") else eps)]
            rows = select_train_rows(self._traj_cache, eps, self.config.hn_p_self,
                                     self.config.hn_context_k, self._traj_gen)
            return rows, [self._traj_cache.key_of_episode(e) for e in eps]
        key = self._eval_key(batch)
        sel = select_eval_rows(self._traj_cache, key, self.config.hn_context_k,
                               self.config.hn_seed)
        bsz = batch[OBS_LANGUAGE_TOKENS].shape[0]
        logger.warning("[TRAJ] key=%s demos=%d", key, len(sel))
        return [sel] * bsz, [key] * bsz

    def _trunk_clips(self, rows: list) -> list:
        """Per sample, per demo: (cached video tokens, one timestamp per 49-token
        unit). The native trunk rebuilds from them the '<t seconds>'-marked video
        prompt Qwen3.5 sees in a chat, so the timestamps must be the strided pairs'
        own; the raw layout carries no timestamps at all."""
        cache = self._traj_cache
        native = getattr(self.config, "hn_trunk_native", True)
        every = int(self.config.hn_trunk_stride)
        fps = float(getattr(self.config, "hn_trunk_fps", 10.0))

        def stamps(r):
            if not native:
                return []
            n = int(cache.records[r].get("src_len", 0))
            if n <= 0:
                raise ValueError(
                    f"cache record for episode {cache.records[r]['episode']} has no "
                    "src_len (built before task keys): the native trunk prompt needs the "
                    "source frame count — rebuild the cache with the current "
                    "build_qwen35_video_cache.py, or set hn_trunk_native=false")
            return pair_timestamps(n, every, fps)

        return [[(cache.read_row(r), stamps(r)) for r in sample] for sample in rows]

    def _trunk_texts(self, keys: list) -> list[str]:
        """Instruction STRING per sample (the trunk has its own tokenizer)."""
        texts = getattr(self._traj_cache, "task_texts", None) or {}
        out = [texts.get(k, "") for k in keys]
        if any(not x for x in out):
            logger.warning("[TRUNK] empty instruction for tasks %s", keys)
        return out

    def _eval_key(self, batch: Dict[str, Tensor]) -> str:
        def _decode() -> str:
            toks = batch.get(OBS_LANGUAGE_TOKENS) if isinstance(batch, dict) else None
            return self._decode_instruction(toks[0]) if toks is not None else ""

        return resolve_eval_key(batch, self._traj_cache, decode_fn=_decode)

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
