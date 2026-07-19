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
    # Zero-init the hypernet's W_up output heads so generated ΔW = 0 at step 0
    # and the policy starts identical to the frozen base (LoRA B=0 convention).
    # False reproduces the legacy behavior (random ΔW injected at init).
    hn_zero_init_up: bool = True

    # --- Vision-conditioning for the hypernetwork --------------------------------
    # Off by default => the hypernetwork is conditioned on text only. Toggle these
    # to also condition LoRA generation on visual context:
    #   VLM vision:        hn_use_vlm_vision=True,  hn_use_dino=False
    #   VLM vision + DINO: hn_use_vlm_vision=True,  hn_use_dino=True
    # `hn_use_dino=True` alone (no VLM vision) is also valid.
    hn_use_vlm_vision: bool = False  # condition HN on the frozen VLM's own image tokens
    hn_use_dino: bool = False        # additionally condition on a frozen external DINOv2
    hn_dino_model_id: str = "facebook/dinov2-base"

    # At EVAL only: cache the generated LoRA adapter once per episode (computed on
    # the first observation after reset) and reuse it for the whole rollout,
    # instead of regenerating it every forward. The task is constant within an
    # episode, so the adapter should be too. This breaks the vision-conditioned
    # feedback loop (live frame -> HN -> LoRA -> action -> next frame) that makes
    # a drifting arm feed itself out-of-distribution LoRA and diverge. No effect
    # during training (the HN must keep regenerating to learn). Env var
    # HN_LORA_CACHE={off,episode} overrides this at runtime.
    hn_lora_cache_eval: bool = False

    # --- Init-frame pairing ablation (vision conditioning) ----------------------------
    # "obs" = condition on the current observation (legacy). "same"/"cross" = at TRAIN
    # time condition on the t=0 frame of the imitated episode / of another same-task
    # episode, read from the first-frame bank. Eval always uses the rollout's own
    # frames (freeze the adapter at t=0 via HN_LORA_CACHE=episode).
    hn_frame_source: str = "obs"
    hn_frame_bank_path: str | None = None
    hn_bank_seed: int = 42
    hn_context_k: int = 8    # cross: max t=0 frames from distinct other episodes
