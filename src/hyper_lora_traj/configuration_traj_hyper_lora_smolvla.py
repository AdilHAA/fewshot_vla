"""Trajectory/video-conditioned Hyper-LoRA — a SEPARATE, backward-compatible policy.

`traj_hyper_lora_smolvla` extends `hyper_lora_smolvla` (frozen SmolVLA + a hypernetwork
that generates standard LoRA (W_down, W_up) for the VLM). It does NOT modify the parent.
With every new knob at its default, this policy is structurally identical to
`hyper_lora_smolvla`.

Trajectory conditioning uses leave-one-out (LOO) task demos: each conditioning
sample draws its context demos from the same task, excluding the query.
Old traj checkpoints are incompatible with this config by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from src.hyper_lora.configuration_hyper_lora_smolvla import HyperLoRASmolVLAConfig


@PreTrainedConfig.register_subclass("traj_hyper_lora_smolvla")
@dataclass
class TrajHyperLoRASmolVLAConfig(HyperLoRASmolVLAConfig):
    # --- Stage 1: LoRA injection sites ------------------------------------------------
    # The parent always patches the VLM text-MLP via `hn_target_module_names`.
    # hn_inject_vlm_mlp=True keeps that exactly (so all-OFF == parent); set False to
    # drop the MLP site (ablation).
    hn_inject_vlm_mlp: bool = True
    # Patch VLM self-attn k_proj/v_proj — the "task key/value supply" the action expert
    # cross-attends to. Computed once and KV-cached -> free at every denoise step.
    hn_inject_vlm_kv: bool = False

    # --- Stage 4 (DEFERRED scaffolding, default-OFF) -----------------------------------
    hn_inject_expert_q: bool = False     # expert q_proj on the cross layers (i % N != 0)

    # --- Trajectory conditioning (leave-one-out task demos) ---------------------------
    hn_use_traj_clip: bool = False
    hn_traj_encoder: str = "dino"        # "dino" (CLS/frame) | "vjepa2" (mean-pool/tubelet)
    hn_xpair_cache_path: str | None = None   # required when hn_use_traj_clip=True
    hn_pair_mode: str = "loo"            # train pairing: "loo" (leave-one-out) | "same"
    hn_context_k: int = 8                # demos per conditioning sample (train U{1..k}; eval k)
    hn_seed: int = 42                    # selector RNG seed (train stream + eval determinism)

    # --- HN fusion extras (neutral => FusionHyperNetwork fast-path == parent) ---------
    hn_stream_type_emb: bool = False
    hn_per_stream_null: bool = False
    hn_readout: str = "queries"          # "queries" (==parent) | "xattn" (DEFERRED)
