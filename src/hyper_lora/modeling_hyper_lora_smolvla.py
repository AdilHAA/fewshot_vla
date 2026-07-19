"""Frozen SmolVLA with hypernetwork-generated LoRA, as a lerobot policy_type.

`__init__` loads the base weights, wraps the VLM MLP linears with
DynamicLoRALinear, builds the hypernet, and freezes everything but the hypernet.
Each forward generates per-sample LoRA from the instruction and injects it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# ImageNet stats for DINOv2 preprocessing (batch images arrive in [0, 1]).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

from .configuration_hyper_lora_smolvla import HyperLoRASmolVLAConfig
from .dynamic_lora import DynamicLoRALinear
from .hypernetwork import HyperNetwork

logger = logging.getLogger(__name__)


class HyperLoRASmolVLAPolicy(SmolVLAPolicy):
    config_class = HyperLoRASmolVLAConfig
    name = "hyper_lora_smolvla"

    def __init__(self, config: HyperLoRASmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Init only; a resumed checkpoint's state_dict overwrites these later.
        if config.base_smolvla_path:
            self._load_base_smolvla_weights(config.base_smolvla_path)

        # Optional external frozen vision encoder (DINOv2), built before the
        # hypernet so its feature dim can size the projection.
        self.dino = None
        dino_dim = 0
        if config.hn_use_dino:
            dino_dim = self._build_dino(config.hn_dino_model_id)

        self._patched: Dict[str, Dict[int, DynamicLoRALinear]] = {}
        target_modules = self._patch_mlp_layers(config)
        self.hypernet = HyperNetwork(
            text_embed_dim=self._vlm_text_hidden_size(),
            hidden_size=config.hn_hidden_size,
            num_layers=len(self._vlm_text_model().layers),
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            dropout=config.hn_dropout,
            encoder_type=config.hn_encoder_type,
            tf_num_blocks=config.hn_tf_num_blocks,
            tf_num_heads=config.hn_tf_num_heads,
            # vision-conditioning: image tokens from the frozen VLM and/or DINO
            use_vlm_vision=config.hn_use_vlm_vision,
            vlm_vision_dim=self._vlm_text_hidden_size(),  # VLM image tokens live in text space
            use_dino=config.hn_use_dino,
            dino_dim=dino_dim,
            zero_init_up=config.hn_zero_init_up,
        )
        self._freeze_base()

        # Init-frame pairing ablation (vision conditioning). Lazily built bank +
        # deterministic generator; validated eagerly so a misconfigured run fails
        # at construction rather than mid-training.
        self._frame_bank = None
        self._bank_gen = torch.Generator().manual_seed(int(getattr(config, "hn_bank_seed", 42)))
        if getattr(config, "hn_frame_source", "obs") != "obs" and not config.hn_frame_bank_path:
            raise ValueError("hn_frame_source != 'obs' requires hn_frame_bank_path")

        self._lora_cache = None
        self._prev_lora_flat = None
        self._lora_step = 0
        self._prev_action = None
        self._act_step = 0

    def reset(self):
        """Called by lerobot at each environment reset. Drop the per-episode
        LoRA cache and drift-logging state so the next episode starts fresh."""
        super().reset()
        self._lora_cache = None
        self._prev_lora_flat = None
        self._lora_step = 0
        self._prev_action = None
        self._act_step = 0

    @torch.no_grad()
    def select_action(self, batch, *args, **kwargs):
        a = super().select_action(batch, *args, **kwargs)
        if os.environ.get("HN_LOG_ACTION"):
            self._log_action(a)
        return a

    def _log_action(self, a: Tensor) -> None:
        """Log executed-action norm, step-to-step jerk, and max abs component — a
        GPU-free behavioral probe for divergence/jitter. Enabled with HN_LOG_ACTION=1."""
        x = a.detach().float()
        flat = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
        prev = self._prev_action
        jerk = (
            (flat - prev).norm(dim=-1).mean().item()
            if prev is not None and prev.shape == flat.shape
            else float("nan")
        )
        self._prev_action = flat
        self._act_step += 1
        logger.warning(
            "[ACT] step=%d a_norm=%.4f jerk=%.4f amax=%.4f",
            self._act_step, flat.norm(dim=-1).mean().item(), jerk, x.abs().max().item(),
        )

    def _load_base_smolvla_weights(self, path: str) -> None:
        base = SmolVLAPolicy.from_pretrained(path)
        missing, unexpected = self.load_state_dict(base.state_dict(), strict=False)
        if missing:
            logger.info(
                "Base SmolVLA load: %d keys not in base (expected — hypernet "
                "etc.): %s%s",
                len(missing), missing[:3], "..." if len(missing) > 3 else "",
            )
        if unexpected:
            logger.warning(
                "Base SmolVLA load: %d unexpected keys: %s%s",
                len(unexpected), unexpected[:3], "..." if len(unexpected) > 3 else "",
            )

    def _save_pretrained(self, save_directory) -> None:
        """Save only the *trainable* weights (hypernet, + action expert if
        unfrozen). lerobot's default writes the whole state_dict, so the frozen
        SmolVLA base (~1.5GB) — and now DINO — would be re-saved every checkpoint.
        The base is reconstructed on load from `base_smolvla_path` and DINO from
        `hn_dino_model_id` in __init__; from_pretrained loads non-strict, so the
        omitted base/DINO keys are simply kept as reconstructed. ~1.5GB -> ~25MB.
        """
        self.config._save_pretrained(save_directory)
        target = self.module if hasattr(self, "module") else self
        trainable = {n for n, p in target.named_parameters() if p.requires_grad}
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in target.state_dict().items()
            if k in trainable
        }
        save_file(state, str(Path(save_directory) / SAFETENSORS_SINGLE_FILE))

    def _vlm_text_model(self) -> nn.Module:
        return self.model.vlm_with_expert.get_vlm_model().text_model

    def _vlm_text_hidden_size(self) -> int:
        cfg = self.model.vlm_with_expert.config
        text_cfg = getattr(cfg, "text_config", cfg)
        return int(text_cfg.hidden_size)

    def _patch_mlp_layers(
        self, config: HyperLoRASmolVLAConfig
    ) -> Dict[str, Tuple[int, int]]:
        text_model = self._vlm_text_model()
        target_modules: Dict[str, Tuple[int, int]] = {}
        for layer_idx, layer in enumerate(text_model.layers):
            mlp = layer.mlp
            for mod_name in config.hn_target_module_names:
                base = getattr(mlp, mod_name)
                if not isinstance(base, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at layer {layer_idx}.mlp.{mod_name}, "
                        f"got {type(base).__name__}"
                    )
                wrapper = DynamicLoRALinear(
                    base_layer=base,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                )
                setattr(mlp, mod_name, wrapper)
                self._patched.setdefault(mod_name, {})[layer_idx] = wrapper
                target_modules[mod_name] = (base.in_features, base.out_features)
        return target_modules

    def _freeze_base(self) -> None:
        """Freeze everything but the hypernet. The loss still backprops into the
        hypernet through the frozen layers' activations."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.hypernet.parameters():
            p.requires_grad = True
        # DINO is a frozen conditioning encoder — keep it frozen and in eval mode
        # even though `.train()` is called on the policy.
        if self.dino is not None:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()
        if getattr(self.config, "train_action_expert", False):
            for p in self.model.vlm_with_expert.lm_expert.parameters():
                p.requires_grad = True

    # --- vision conditioning -------------------------------------------------
    def _build_dino(self, model_id: str) -> int:
        """Load a frozen external DINOv2 and register ImageNet norm buffers.
        Returns the DINO feature dim (to size the hypernet projection)."""
        from transformers import AutoModel

        self.dino = AutoModel.from_pretrained(model_id)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        self.register_buffer(
            "_dino_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_dino_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        return int(self.dino.config.hidden_size)

    def _first_image(self, batch: Dict[str, Tensor]) -> Tensor:
        """The main camera frame from the batch, as (B, C, H, W) in [0, 1]."""
        key = next(k for k in self.config.image_features if k in batch)
        img = batch[key]
        if img.ndim == 5:  # (B, T, C, H, W) -> last frame
            img = img[:, -1]
        return img

    def _vlm_vision_features(self, batch: Dict[str, Tensor]) -> Tensor:
        """Image tokens from the *frozen VLM's own* SigLIP encoder.
        Returns (B, sum_tokens, vlm_hidden). Computed without grad (frozen)."""
        images, _img_masks = self.prepare_images(batch)
        toks = [self.model.vlm_with_expert.embed_image(img) for img in images]
        return torch.cat(toks, dim=1)

    def _dino_features(self, batch: Dict[str, Tensor]) -> Tensor:
        """Patch tokens from the external frozen DINOv2 on the main frame.
        Returns (B, 1 + num_patches, dino_hidden). Computed without grad."""
        img = self._first_image(batch).to(self._dino_mean.dtype)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self._dino_mean) / self._dino_std
        out = self.dino(pixel_values=img.to(self.dino.dtype))
        return out.last_hidden_state

    def _embed_language(self, lang_tokens: Tensor) -> Tensor:
        return self.model.vlm_with_expert.embed_language_tokens(lang_tokens)

    def _lora_cache_mode(self) -> str:
        """'episode' = generate the adapter once per rollout and reuse it; 'off'
        = regenerate every forward (legacy). Env var HN_LORA_CACHE wins over the
        config default. Never caches during training."""
        env = os.environ.get("HN_LORA_CACHE")
        if env:
            return env.strip().lower()
        return "episode" if getattr(self.config, "hn_lora_cache_eval", False) else "off"

    def _log_lora_drift(self, weights) -> None:
        """Log L2 norm of all generated LoRA params and the L2 change since the
        previous forward — a cheap probe for the vision-conditioned feedback
        loop. Enabled with env var HN_LOG_LORA=1."""
        flat = torch.cat(
            [t.reshape(-1) for layers in weights.values() for pair in layers.values() for t in pair]
        ).float()
        prev = self._prev_lora_flat
        drift = (
            (flat - prev).norm().item()
            if prev is not None and prev.numel() == flat.numel()
            else float("nan")
        )
        self._prev_lora_flat = flat.detach()
        self._lora_step += 1
        logger.warning(
            "[HN] step=%d lora_norm=%.4f step_drift=%.4f", self._lora_step, flat.norm().item(), drift
        )

    def _ensure_bank(self) -> None:
        if self._frame_bank is None:
            from src.traj_data.frame_bank import FrameBank

            self._frame_bank = FrameBank(self.config.hn_frame_bank_path)

    def _swap_main(self, batch: Dict[str, Tensor], img: Tensor) -> Dict[str, Tensor]:
        """Batch copy with the MAIN camera replaced by `img` (B,C,H,W). Other views
        and text/action keys stay real: the bank stores agentview frames only, and
        duplicating one into the wrist slot would give the conditioning encoders a
        train-time input distribution that never occurs at eval."""
        out = dict(batch)
        key = next(k for k in self.config.image_features if k in out)  # == _first_image
        out[key] = img.to(next(self.parameters()).device)
        return out

    def _bank_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """'same' conditioning: the t=0 frame of the imitated episode itself."""
        self._ensure_bank()
        frames = [self._frame_bank.same(int(ep), self._bank_gen)
                  for ep in batch["episode_index"].tolist()]
        return self._swap_main(batch, torch.stack(frames))

    def _bank_frame_sets(self, batch: Dict[str, Tensor]) -> list:
        """'cross' conditioning: ALWAYS hn_context_k t=0 frames from DISTINCT other
        episodes of the same task ("frames of different trajectories"); fewer only
        when some task in the batch has a smaller pool (no repeat-padding). One
        k_step per batch keeps the vision-token count uniform across samples.
        Returns k_step stacks of (B,C,H,W)."""
        self._ensure_bank()
        kmax = int(getattr(self.config, "hn_context_k", 8))
        pairs = list(zip(batch["episode_index"].tolist(), batch["task_index"].tolist()))
        avail = [self._frame_bank.n_cross(int(t), int(ep)) for ep, t in pairs]
        k_step = (min([kmax] + [a for a in avail if a > 0])
                  if any(a > 0 for a in avail) else 1)
        per_sample = [self._frame_bank.cross_set(int(t), int(ep), self._bank_gen, k_step)
                      for ep, t in pairs]
        return [torch.stack([s[i] for s in per_sample]) for i in range(k_step)]

    def _main_token_slice(self, toks: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Tokens of the main camera only (first in prepare_images order): extra bank
        frames must not re-contribute duplicate wrist tokens."""
        n_cams = sum(1 for k in self.config.image_features if k in batch)
        return toks[:, : toks.shape[1] // max(n_cams, 1)]

    def _bank_eval_sets(self, batch: Dict[str, Tensor]):
        """Eval few-shot context for 'cross': resolve the task (batch task_index, then
        the instruction string) and return hn_context_k deterministic bank frames of
        its train episodes, expanded across envs. None when unresolvable."""
        self._ensure_bank()
        ti = batch.get("task_index") if isinstance(batch, dict) else None
        if ti is not None:
            ti = int(ti.flatten()[0]) if hasattr(ti, "flatten") else int(ti)
        else:
            text = ""
            for key in ("task", "instruction", "language_instruction", "prompt"):
                t = batch.get(key) if isinstance(batch, dict) else None
                if isinstance(t, (list, tuple)) and t:
                    t = t[0]
                if isinstance(t, str) and t:
                    text = t
                    break
            ti = self._frame_bank.resolve_task(text) if text else None
        if ti is None:
            return None
        frames = self._frame_bank.task_set(
            ti, int(getattr(self.config, "hn_context_k", 8)),
            int(getattr(self.config, "hn_bank_seed", 42)))
        if not frames:
            return None
        bsz = batch[OBS_LANGUAGE_TOKENS].shape[0]
        dev = next(self.parameters()).device
        return [f.unsqueeze(0).expand(bsz, -1, -1, -1).to(dev) for f in frames]

    def _inject_lora(self, batch: Dict[str, Tensor]) -> None:
        # At eval, optionally reuse a per-episode adapter to break the
        # vision-conditioned feedback loop (see config.hn_lora_cache_eval).
        cache_episode = (not self.training) and self._lora_cache_mode() == "episode"
        if cache_episode and self._lora_cache is not None:
            self._set_lora_weights(self._lora_cache)
            return

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.set_grad_enabled(self.training):
            text_embeds = self._embed_language(lang_tokens)

        # Init-frame pairing ablation: at TRAIN time swap the conditioning images.
        # 'same' = the imitated episode's own t=0 frame; 'cross' = k t=0 frames from
        # distinct other episodes of the task. The default "obs" path leaves
        # cond_batch == batch, so vision behavior is bit-for-bit unchanged.
        cond_batch = batch
        extra_frames = []
        rep_k = 0
        if getattr(self.config, "hn_frame_source", "obs") == "cross" and not self.training:
            # cross-trained HN always consumed hn_context_k frames. Preferred: resolve
            # the task and feed k bank frames of its TRAIN episodes (format == train).
            # Fallback (bank has no task_texts / unresolvable): replicate the rollout
            # frame to match the train-time token count.
            sets = self._bank_eval_sets(batch)
            if sets:
                cond_batch = self._swap_main(batch, sets[0])
                extra_frames = sets[1:]
            else:
                rep_k = int(getattr(self.config, "hn_context_k", 8)) - 1
        if self.training and getattr(self.config, "hn_frame_source", "obs") != "obs":
            if self.config.hn_frame_source == "cross":
                sets = self._bank_frame_sets(batch)
                cond_batch = self._swap_main(batch, sets[0])
                extra_frames = sets[1:]
            else:
                cond_batch = self._bank_batch(batch)

        # Vision conditioning inputs come from frozen encoders, so compute them
        # under no_grad (cheaper); the hypernet still trains via the LoRA path.
        # Frames beyond the first contribute only their main-camera tokens.
        vlm_vision_embeds = None
        dino_embeds = None
        with torch.no_grad():
            if self.config.hn_use_vlm_vision:
                vlm_vision_embeds = self._vlm_vision_features(cond_batch)
                if extra_frames:
                    extras = [self._main_token_slice(
                        self._vlm_vision_features(self._swap_main(batch, f)), batch)
                        for f in extra_frames]
                    vlm_vision_embeds = torch.cat([vlm_vision_embeds] + extras, dim=1)
                elif rep_k > 0:
                    ms = self._main_token_slice(vlm_vision_embeds, batch)
                    vlm_vision_embeds = torch.cat([vlm_vision_embeds] + [ms] * rep_k, dim=1)
            if self.config.hn_use_dino:
                dino_embeds = self._dino_features(cond_batch)
                if extra_frames:
                    extras = [self._dino_features(self._swap_main(batch, f))
                              for f in extra_frames]
                    dino_embeds = torch.cat([dino_embeds] + extras, dim=1)
                elif rep_k > 0:
                    dino_embeds = torch.cat([dino_embeds] * (rep_k + 1), dim=1)

        weights = self.hypernet(
            text_embeds, lang_masks, vlm_vision_embeds, dino_embeds
        )
        if os.environ.get("HN_LOG_LORA"):
            self._log_lora_drift(weights)
        self._set_lora_weights(weights)
        if cache_episode:
            self._lora_cache = weights

    def _set_lora_weights(
        self, weights: Dict[str, Dict[int, Tuple[Tensor, Tensor]]]
    ) -> None:
        for mod_name, layers in self._patched.items():
            mod_weights = weights.get(mod_name, {})
            for layer_idx, wrapper in layers.items():
                pair = mod_weights.get(layer_idx)
                if pair is not None:
                    wrapper.set_lora_weights(*pair)

    def _clear_lora(self) -> None:
        for layers in self._patched.values():
            for wrapper in layers.values():
                wrapper.clear_lora_weights()

    def forward(self, batch, **kwargs):
        self._inject_lora(batch)
        try:
            return super().forward(batch, **kwargs)
        finally:
            self._clear_lora()

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        self._inject_lora(batch)
        try:
            return super()._get_action_chunk(batch, noise=noise, **kwargs)
        finally:
            self._clear_lora()

    def trainable_parameter_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "hypernet": count(self.hypernet),
            "lm_expert": count(self.model.vlm_with_expert.lm_expert),
            "vlm": count(self.model.vlm_with_expert.vlm),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
