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

    # --- Trajectory conditioning (within-task demo pairing) ----------------------------
    hn_use_traj_clip: bool = False
    hn_traj_encoder: str = "dino"        # "dino" (CLS/frame) | "vjepa2" (tubelet grid-pool)
    hn_xpair_cache_path: str | None = None   # required when hn_use_traj_clip=True
    hn_p_self: float = 0.0               # P(context = imitated episode itself); 0.0 = cross
    hn_context_k: int = 1                # demos per conditioning sample (train and eval)
    hn_vjepa_grid: int = 2               # vjepa2: s×s spatial tokens per tubelet (1 = mean-pool)
    hn_seed: int = 42                    # selector RNG seed (train stream + eval determinism)
    # Temporal scope of a cached record. Defaults reproduce the full-clip arms
    # exactly; time_select="first" is the minimal-clip arm (n_frames=1, fill="dup"
    # -> one frame duplicated into a single tubelet -> one token).
    hn_traj_time_select: str = "all"     # "all" | "first"
    hn_traj_n_frames: int = 0            # frames per record when time_select="first"
    hn_traj_fill: str = ""               # "" | "dup" | "zero"
    # Opt-in provenance pins (empty/zero = don't check). `encoder_id` alone does
    # not identify the weights, and the encode window is invisible in the tokens.
    hn_traj_encoder_model: str = ""      # HF id of the encoder that built the cache
    hn_traj_chunk: int = 0               # encode window the cache was built with
    # Which DINO tokens the cache holds per frame (vjepa2 uses hn_vjepa_grid).
    # grid=16 at 224px is the IDENTITY pool, i.e. all 256 patches raw.
    hn_dino_grid: int = 0                # 0 = no patch tokens
    hn_dino_n_reg: int = 0               # register tokens (dinov2-with-registers only)
    hn_dino_include_cls: bool = True
    # --- temporal / sub-token position code for the traj stream ----------------------
    # The HN has no positional encoding of its own and its output is provably
    # permutation-invariant, so without this the clip is a BAG of frames.
    hn_traj_pos_emb: str = "none"        # "none" | "index" | "phase"
    hn_traj_max_pos: int = 512           # table rows; max LIBERO episode is 505 frames
    hn_traj_sub_emb: bool = False        # index of the token WITHIN its frame/tubelet
    hn_traj_pos_std: float = 0.02        # init std; fixed so index/phase start equal

    # --- Direction 7: pretrained trunk instead of the scratch hypernetwork ----------
    # Empty (default) => FusionHyperNetwork exactly as before; every existing arm,
    # command and checkpoint is untouched. Set to a Qwen3.5 HF id to condition via
    # its frozen text stack over CACHED video tokens + instruction + 32 layer tokens.
    hn_trunk_model: str = ""
    hn_trunk_text: bool = True       # include the instruction in the trunk input
    hn_trunk_grad_ckpt: bool = True  # required: activations for the layer tokens
    hn_trunk_stride: int = 4         # keep every k-th frame (fixed interval)

    # --- HN fusion extras (neutral => FusionHyperNetwork fast-path == parent) ---------
    hn_stream_type_emb: bool = False
    hn_per_stream_null: bool = False
    hn_readout: str = "queries"          # "queries" (==parent) | "xattn" (DEFERRED)
