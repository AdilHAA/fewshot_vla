"""Frozen SmolVLA with hypernetwork-generated LoRA, as a lerobot policy_type.

`__init__` loads the base weights, wraps the VLM MLP linears with
DynamicLoRALinear, builds the hypernet, and freezes everything but the hypernet.
Each forward generates per-sample LoRA from the instruction and injects it.
"""

from __future__ import annotations

import logging
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
        )
        self._freeze_base()

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

    def _inject_lora(self, batch: Dict[str, Tensor]) -> None:
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.set_grad_enabled(self.training):
            text_embeds = self._embed_language(lang_tokens)

        # Vision conditioning inputs come from frozen encoders, so compute them
        # under no_grad (cheaper); the hypernet still trains via the LoRA path.
        vlm_vision_embeds = None
        dino_embeds = None
        with torch.no_grad():
            if self.config.hn_use_vlm_vision:
                vlm_vision_embeds = self._vlm_vision_features(batch)
            if self.config.hn_use_dino:
                dino_embeds = self._dino_features(batch)

        weights = self.hypernet(
            text_embeds, lang_masks, vlm_vision_embeds, dino_embeds
        )
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
