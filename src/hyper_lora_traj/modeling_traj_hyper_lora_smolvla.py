"""TrajHyperLoRASmolVLAPolicy — trajectory/video-conditioned Hyper-LoRA policy.

Backward-compatible subclass of HyperLoRASmolVLAPolicy. With all new flags at their
defaults (hn_use_traj_clip=False), the policy is structurally identical to the parent:
FusionHyperNetwork builds no extra params and _inject_lora passes no traj_embeds.
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
from src.traj_data.encoder import build_traj_encoder
from src.traj_data.xpair_select import select_eval_conditioning, select_train_conditioning

from .configuration_traj_hyper_lora_smolvla import TrajHyperLoRASmolVLAConfig
from .fusion_hypernetwork import FusionHyperNetwork

logger = logging.getLogger(__name__)


class TrajHyperLoRASmolVLAPolicy(HyperLoRASmolVLAPolicy):
    config_class = TrajHyperLoRASmolVLAConfig
    name = "traj_hyper_lora_smolvla"

    def __init__(self, config: TrajHyperLoRASmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Open the offline trajectory cache; traj feature dim comes from its header.
        self._traj_cache = None
        traj_dim = 0
        perceiver_msl = 8192
        if config.hn_use_traj_clip:
            if not config.hn_xpair_cache_path:
                raise ValueError("hn_use_traj_clip=True requires hn_xpair_cache_path")
            from src.traj_data.traj_cache import TrajCache

            self._traj_cache = TrajCache(config.hn_xpair_cache_path)
            self._traj_cache.assert_header_matches(
                num_frames=config.hn_traj_num_frames, encoder_id=config.hn_traj_encoder)
            hdr = self._traj_cache.header
            traj_dim = int(hdr["d_enc"])
            # Perceiver pos-emb must cover the longest demo concat (retrieval k or
            # train k clips); e.g. V-JEPA2 clips are 2048 tokens -> k=6 needs 12288.
            n_tok = int(hdr.get("n_tokens") or hdr["num_frames"] * hdr["ntok"])
            k_max = max(config.hn_retrieval_k, getattr(config, "hn_train_k", 1), 1)
            perceiver_msl = max(8192, k_max * n_tok)

        # Replace the plain HyperNetwork with FusionHyperNetwork over the same
        # target_modules. With all fusion flags off, params/forward are identical.
        tm = self.hypernet.target_modules
        dino_dim = (
            int(self.dino.config.hidden_size) if getattr(self, "dino", None) is not None else 0
        )
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
            perceiver_latents=config.hn_perceiver_latents,
            perceiver_depth=config.hn_perceiver_depth,
            perceiver_heads=config.hn_perceiver_heads,
            perceiver_max_seq_len=perceiver_msl,
        )
        self._freeze_base()  # re-apply requires_grad to the new (Fusion) hypernet

        self._traj_gen = torch.Generator().manual_seed(int(config.hn_seed))
        # Eval-only live conditioning, built lazily on first eval use.
        self._frame_buf = None
        self._traj_encoder = None
        self._traj_encode = None
        self._sent_enc = None
        if config.hn_use_traj_clip:
            from src.traj_data.frame_buffer import FrameBuffer

            self._frame_buf = FrameBuffer(config.hn_traj_num_frames, config.hn_traj_frame_stride)

    # --- backward-compatible site patching -------------------------------------------
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

        traj_embeds = None
        traj_mask = None
        if getattr(self.config, "hn_use_traj_clip", False):
            with torch.no_grad():
                traj_embeds, traj_mask = self._build_traj_conditioning(batch)

        weights = self.hypernet(
            text_embeds, lang_masks, vlm_vision_embeds, dino_embeds,
            traj_embeds=traj_embeds, traj_mask=traj_mask,
        )
        if os.environ.get("HN_LOG_LORA"):
            self._log_lora_drift(weights)
        self._set_lora_weights(weights)
        if cache_episode:
            self._lora_cache = weights

    def _hypernet_device(self) -> torch.device:
        return next(self.hypernet.parameters()).device

    def _build_traj_conditioning(self, batch: Dict[str, Tensor]):
        if self.training:
            ep = batch["episode_index"]
            t = batch["task_index"]
            ep = ep.tolist() if hasattr(ep, "tolist") else list(ep)
            t = t.tolist() if hasattr(t, "tolist") else list(t)
            traj, mask = select_train_conditioning(
                self._traj_cache, ep, t, self.config.hn_p_self, self._traj_gen,
                k=getattr(self.config, "hn_train_k", 1),
            )
            return traj.to(self._hypernet_device()), mask
        return self._build_eval_conditioning(batch)

    # --- eval live conditioning ------------------------------------------------------
    def _ensure_eval_encoders(self) -> None:
        """Lazily build the frozen clip encoder (dino/vjepa2) + sentence encoder for
        retrieval, on first eval use so training never pays for them."""
        if self._traj_encode is None:
            model_id = (self.config.hn_dino_model_id if self.config.hn_traj_encoder == "dino"
                        else self.config.hn_vjepa_model_id)
            dev = next(self.parameters()).device
            self._traj_encoder, self._traj_encode = build_traj_encoder(
                self.config.hn_traj_encoder, model_id, dev)
        if self._sent_enc is None and self.config.hn_traj_source == "retrieval":
            from sentence_transformers import SentenceTransformer

            sid = self._traj_cache.header.get(
                "sentence_encoder_id", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._sent_enc = SentenceTransformer(sid)

    def _build_eval_conditioning(self, batch: Dict[str, Tensor]):
        # Eval envs are vectorized: the buffer holds (B,C,H,W) frames per step, so the
        # conditioning batch must match the observation batch.
        self._ensure_eval_encoders()
        dev = next(self._traj_encoder.parameters()).device
        clips = self._frame_buf.clip().to(dev).transpose(0, 1)  # (T,B,..) -> (B,T,C,H,W)
        traj_q = self._traj_encode(clips)                        # (B, N, D)
        bsz = traj_q.shape[0]
        hdev = self._hypernet_device()
        source = getattr(self.config, "hn_traj_source", "self")
        if source == "self":
            return traj_q.to(hdev), None

        # task_index is only needed to pick a same-task demo (one_shot) and for
        # provenance labels; retrieval itself queries by [text ⊕ frames] and must
        # NOT be gated on it (rollout batches usually lack task_index).
        task_index = self._eval_task_index(batch)

        if source == "one_shot":
            if task_index is None:  # cannot pick a same-task demo -> self fallback
                return traj_q.to(hdev), None
            eps = self._traj_cache.episodes_of_task(task_index)
            row = self._traj_cache._row.get((eps[0], 0)) if eps else None
            if row is None:
                return traj_q.to(hdev), None
            demo = self._traj_cache.read_rows([row]).reshape(1, -1, traj_q.shape[-1])
            return demo.expand(bsz, -1, -1).to(hdev), None

        # retrieval (headline): per-env query; kill-switch fallbacks make lengths ragged
        # across the batch -> pad to the longest and mask (True == ignore).
        text_emb = self._eval_text_emb(batch)
        frame_embs = traj_q.float().mean(dim=1)                  # (B, D)
        prov_task = -1 if task_index is None else task_index     # -1: provenance unknown
        parts, fallbacks, prov = [], 0, []
        for b in range(bsz):
            traj_b, _, prov_b, fb = select_eval_conditioning(
                self._traj_cache, traj_q[b], frame_embs[b], text_emb.to(frame_embs),
                self.config.hn_retrieval_k, self.config.hn_retrieval_tau, prov_task,
                self.config.hn_retrieval_beta_t, self.config.hn_retrieval_beta_f,
            )
            parts.append(traj_b[0])                              # (L_b, D)
            fallbacks += int(fb)
            prov = prov_b
        lmax = max(p.shape[0] for p in parts)
        traj = traj_q.new_zeros(bsz, lmax, traj_q.shape[-1])
        mask = torch.ones(bsz, lmax, dtype=torch.bool, device=traj.device)
        for b, p in enumerate(parts):
            traj[b, : p.shape[0]] = p.to(traj)
            mask[b, : p.shape[0]] = False
        logger.warning("[RETRIEVAL] task=%s envs=%d fallbacks=%d provenance=%s",
                       task_index, bsz, fallbacks, prov)
        return traj.to(hdev), mask.to(hdev)

    def _eval_task_index(self, batch: Dict[str, Tensor]):
        # eval/rollout batches come from the env, not LeRobotDataset; task_index may be
        # absent — falls back to self conditioning when unavailable.
        v = batch.get("task_index") if isinstance(batch, dict) else None
        if v is None:
            return None
        return int(v.flatten()[0]) if hasattr(v, "flatten") else int(v)

    def _eval_text_emb(self, batch: Dict[str, Tensor]) -> Tensor:
        """Sentence embedding of the instruction for the retrieval text key.
        Decodes OBS_LANGUAGE_TOKENS via the VLM tokenizer when available."""
        text = ""
        tok = getattr(getattr(self, "language_tokenizer", None), "decode", None)
        if tok is not None and OBS_LANGUAGE_TOKENS in batch:
            ids = batch[OBS_LANGUAGE_TOKENS][0]
            text = self.language_tokenizer.decode(ids, skip_special_tokens=True)
        emb = self._sent_enc.encode(text)
        return torch.as_tensor(emb, dtype=torch.float32)

    # --- rollout hooks ---------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch, *args, **kwargs):
        if self._frame_buf is not None:
            # Full env batch: eval envs are vectorized, conditioning is per-env.
            self._frame_buf.push(self._first_image(batch).detach())
        return super().select_action(batch, *args, **kwargs)

    def reset(self):
        # SmolVLAPolicy.__init__ calls reset() before our __init__ sets _frame_buf.
        super().reset()
        fb = getattr(self, "_frame_buf", None)
        if fb is not None:
            fb.reset()
