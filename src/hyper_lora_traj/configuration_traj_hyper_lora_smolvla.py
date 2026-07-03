"""Config for TrajHyperLoRASmolVLAPolicy — extends HyperLoRASmolVLAConfig with
trajectory-clip conditioning, extra LoRA injection sites, and auxiliary loss knobs.
All new flags default to OFF so the policy is structurally identical to the parent.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from src.hyper_lora.configuration_hyper_lora_smolvla import HyperLoRASmolVLAConfig


@PreTrainedConfig.register_subclass("traj_hyper_lora_smolvla")
@dataclass
class TrajHyperLoRASmolVLAConfig(HyperLoRASmolVLAConfig):
    # LoRA injection sites
    # hn_inject_vlm_mlp=True preserves the parent MLP site exactly (all-OFF == parent).
    hn_inject_vlm_mlp: bool = True
    # Patch VLM self-attn k/v — the task key/value the action expert cross-attends to.
    hn_inject_vlm_kv: bool = False
    hn_inject_expert_q: bool = False      # expert q_proj on cross-attn layers

    # Trajectory clip conditioning (OFF => single live frame, parent behaviour)
    hn_use_traj_clip: bool = False
    hn_traj_encoder: str = "dino"         # "dino" | "vjepa2"
    hn_vjepa_model_id: str = "facebook/vjepa2-vitl-fpc64-256"  # used when encoder="vjepa2"
    hn_traj_num_frames: int = 4           # vjepa2: use 16 (even; tubelet_size=2)
    hn_traj_frame_stride: int = 1

    # HN fusion extras (neutral => FusionHyperNetwork fast-path == parent)
    hn_stream_type_emb: bool = False
    hn_per_stream_null: bool = False
    hn_readout: str = "queries"           # "queries" (==parent) | "xattn"

    # Perceiver pooler knobs (active only when hn_use_traj_clip=True)
    hn_perceiver_latents: int = 8
    hn_perceiver_depth: int = 2
    hn_perceiver_heads: int = 4

    # Trajectory source: train uses "self_sametask"; eval "retrieval" | "self" | "one_shot"
    hn_traj_source: str = "self_sametask"
    hn_p_self: float = 0.5                # P(self) in train self/same-task mix
    hn_train_k: int = 1                   # concat up to k same-task demos per train sample
    hn_xpair_cache_path: str | None = None  # required when hn_use_traj_clip=True
    hn_retrieval_k: int = 6
    hn_retrieval_tau: float = 0.5         # cosine kill-switch threshold
    hn_retrieval_beta_t: float = 1.0      # text-block weight in retrieval key
    hn_retrieval_beta_f: float = 1.0      # frame-block weight in retrieval key
    hn_seed: int = 42                     # xpair selector RNG seed

    # Auxiliary forcing losses (0.0 => pure flow-matching)
    hn_loss_supcon: float = 0.0
    hn_loss_cf: float = 0.0
    hn_loss_inv: float = 0.0
    hn_loss_vicreg: float = 0.0
    hn_cfg_drop: float = 0.0
