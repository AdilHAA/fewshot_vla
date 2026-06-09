"""SmolVLAConfig + hypernetwork knobs, registered as policy_type
`hyper_lora_smolvla` for the stock lerobot CLI.

The base is the LIBERO-finetuned `HuggingFaceVLA/smolvla_libero`, frozen
end-to-end; only the hypernet trains. When training from scratch the SmolVLA
fields must match the base checkpoint (its narrow expert: expert_width_multiplier
=0.5); train_hyper_lora.py injects them automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("hyper_lora_smolvla")
@dataclass
class HyperLoRASmolVLAConfig(SmolVLAConfig):
    # Base weights loaded on fresh init (a saved checkpoint's state_dict
    # overrides this on resume/eval). None to start from random init.
    base_smolvla_path: str | None = "HuggingFaceVLA/smolvla_libero"

    # True also fine-tunes the action expert; off keeps the base
    # fully frozen so only the hypernet learns.
    train_action_expert: bool = False

    # LoRA / hypernetwork knobs.
    lora_rank: int = 4
    lora_alpha: int = 16
    hn_hidden_size: int = 128
    hn_dropout: float = 0.1
    hn_encoder_type: str = "tf"  # 'tf' or 'mlp'
    hn_tf_num_blocks: int = 2
    hn_tf_num_heads: int = 4
    hn_target_module_names: tuple[str, ...] = field(
        default_factory=lambda: ("gate_proj", "up_proj", "down_proj")
    )

    # --- Vision-conditioning for the hypernetwork --------------------------------
    # Off by default => the hypernetwork is conditioned on text only. Toggle these
    # to also condition LoRA generation on visual context:
    #   VLM vision:        hn_use_vlm_vision=True,  hn_use_dino=False
    #   VLM vision + DINO: hn_use_vlm_vision=True,  hn_use_dino=True
    # `hn_use_dino=True` alone (no VLM vision) is also valid.
    hn_use_vlm_vision: bool = False  # condition HN on the frozen VLM's own image tokens
    hn_use_dino: bool = False        # additionally condition on a frozen external DINOv2
    hn_dino_model_id: str = "facebook/dinov2-base"
