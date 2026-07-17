#!/usr/bin/env bash
# Train: hypernetwork (text or text+vision conditioning) or the stock lerobot
# PEFT-LoRA baseline, on ALL of lerobot/libero (40 tasks, 1693 episodes).
# The SmolVLA base stays frozen; in HN modes only the hypernet trains, in
# `lora` mode only the PEFT adapters do.
#
#   bash scripts/train.sh                       # vision-conditioned hypernet
#   MODE=text bash scripts/train.sh             # text-only hypernet (ablation)
#   MODE=lora bash scripts/train.sh             # stock PEFT LoRA, same sites
#   MODE=traj bash scripts/train.sh             # trajectory-conditioned hypernet
#   RESUME=1 bash scripts/train.sh              # resume OUTPUT from last ckpt
#
# Env vars:
#   MODE       (vision)               vision | text | lora | traj
#   RESUME     (0)                    1 = resume $OUTPUT from its last checkpoint
#   STEPS      (100000)               training steps
#   SEED       (42)                   training seed; non-default seeds get a
#                                     suffixed OUTPUT dir automatically
#   AUG        (0)                    1 = enable dataset image augmentations
#   EXPERT     (0)                    1 = also unfreeze + train the action
#                                     expert (HN modes only; checkpoint grows)
#   RANK       (4)                    LoRA rank (HN generation / PEFT adapter)
#   BATCH      (16 vision, 32 other)  train batch size
#   WORKERS    (8)                    dataloader workers
#   SAVE_FREQ  (25000)                checkpoint interval (steps)
#   OUTPUT     (outputs/<mode>…)      output dir (must not pre-exist unless RESUME=1)
#   PREC       (bf16)                 bf16 | no (fp32)
#   DINO_ID    (facebook/dinov2-base) external vision encoder (vision mode)
#   ENC        (dino)                traj clip encoder: dino | vjepa2
#   XPAIR_CACHE (outputs/xpair_cache/$ENC)  traj-clip cache dir (traj mode; build
#                                     it first with scripts/build_xpair_cache.py
#                                     --encoder $ENC)
#   KV         (0)                    1 = also inject LoRA at the VLM k/v routing
#                                     site (traj mode)
#   PAIR       (loo)                 conditioning pair mode, shared by two modes:
#                                     traj (->hn_pair_mode):    same | loo (leave-one-out)
#                                     vision (->hn_frame_source): obs | same | cross.
#                                     Unset (or the traj default "loo") maps to obs =
#                                     legacy current-observation conditioning.
#   K          (8)                   traj mode: number of context demos per sample
#   BANK       (outputs/frame_bank.npz)  vision mode, PAIR=same|cross: first-frame
#                                     bank path (build with the frame-bank script)
#   VLM        (1)                   vision mode: 1 = condition HN on the VLM's own
#                                     image tokens; 0 = no VLA embedding (arm A3/A4)
#   WANDB      (1)                    1 = enable wandb logging
#   WANDB_PROJECT (hyper-lora)        wandb project name
#   WANDB_OFFLINE (0)                 1 = log wandb locally without network/login
#                                     (later: `wandb sync wandb/offline-*`)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

MODE="${MODE:-vision}"
RESUME="${RESUME:-0}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
AUG="${AUG:-0}"
EXPERT="${EXPERT:-0}"
RANK="${RANK:-4}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
PREC="${PREC:-bf16}"
DINO_ID="${DINO_ID:-facebook/dinov2-base}"
ENC="${ENC:-dino}"
XPAIR_CACHE="${XPAIR_CACHE:-outputs/xpair_cache/$ENC}"
KV="${KV:-0}"
PAIR="${PAIR:-loo}"
K="${K:-8}"
BANK="${BANK:-outputs/frame_bank.npz}"
VLM="${VLM:-1}"
[ "$KV" = "1" ] && KV_FLAG=true || KV_FLAG=false
export ACCELERATE_MIXED_PRECISION="$PREC"
[ "${WANDB:-1}" = "1" ] && WANDB_FLAG=true || WANDB_FLAG=false
[ "$AUG" = "1" ] && AUG_FLAG=true || AUG_FLAG=false
[ "$EXPERT" = "1" ] && EXPERT_FLAG=true || EXPERT_FLAG=false
WANDB_PROJECT="${WANDB_PROJECT:-hyper-lora}"
[ "${WANDB_OFFLINE:-0}" = "1" ] && export WANDB_MODE=offline

case "$MODE" in
    vision) DEFAULT_OUTPUT="outputs/hyper_lora_vision"; BATCH="${BATCH:-16}" ;;
    text)   DEFAULT_OUTPUT="outputs/hyper_lora_text";   BATCH="${BATCH:-32}" ;;
    lora)   DEFAULT_OUTPUT="outputs/lora_baseline";     BATCH="${BATCH:-32}" ;;
    traj)   DEFAULT_OUTPUT="outputs/hyper_lora_traj_$ENC"; BATCH="${BATCH:-32}" ;;
    *) echo "ERROR: MODE must be vision | text | lora | traj, got '$MODE'" >&2; exit 1 ;;
esac
# Ablation toggles get distinct default output dirs so runs don't collide.
[ "$RANK" != "4" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_r${RANK}"
[ "$SEED" != "42" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_s${SEED}"
[ "$AUG" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_aug"
[ "$EXPERT" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_expert"
[ "$MODE" = "traj" ] && [ "$KV" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_kv"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"

if [ "$RESUME" = "1" ]; then
    CONFIG="$OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: $CONFIG not found — nothing to resume." >&2
        exit 1
    fi
    echo "==> Resuming $OUTPUT from $(readlink -f "$OUTPUT/checkpoints/last") | save_freq=$SAVE_FREQ"
    exec python train_hyper_lora.py --config_path="$CONFIG" --resume=true --save_freq="$SAVE_FREQ"
fi

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output dir '$OUTPUT' already exists (lerobot refuses to overwrite)."
    echo "       Set OUTPUT=… to a fresh path, remove it, or pass RESUME=1."
    exit 1
fi

# Work around lerobot/libero's wrong meta/episodes file_index (idempotent, ~21MB):
# pre-fetch every data parquet so the loader globs + filters by episode_index.
python -c "from src.data.libero import prefetch_all_data_parquets as p; p()"

# Mode-specific policy flags.
MODE_ARGS=()
case "$MODE" in
    vision)
        # PAIR serves both modes: for vision, obs (legacy) is the default, so an
        # unset PAIR (or the traj default "loo") maps to obs. PAIR=same|cross pick
        # the init-frame ablation arms; only those pass a frame-bank path.
        vpair=$PAIR
        [ "$vpair" = "loo" ] && vpair=obs
        MODE_ARGS+=(
            --policy.type=hyper_lora_smolvla
            --policy.hn_use_vlm_vision=$([ "$VLM" = "1" ] && echo true || echo false)
            --policy.hn_frame_source="$vpair"
            --policy.hn_use_dino=true
            --policy.hn_dino_model_id="$DINO_ID"
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        )
        [ "$vpair" != "obs" ] && MODE_ARGS+=(--policy.hn_frame_bank_path="$BANK") ;;
    text)
        MODE_ARGS+=(
            --policy.type=hyper_lora_smolvla
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        ) ;;
    lora)
        # Stock lerobot PEFT LoRA on the same injection sites as the hypernet
        # (VLM text_model MLP linears). The trainer wraps the policy with PEFT
        # when --peft.* is given; the checkpoint saves a standard adapter that
        # lerobot-eval loads automatically (config.use_peft=true).
        MODE_ARGS+=(
            --policy.path=HuggingFaceVLA/smolvla_libero
            --peft.r="$RANK"
            --peft.target_modules='model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)'
        ) ;;
    traj)
        # Trajectory-conditioned hypernet: the HN reads a demo clip from the
        # offline cache. Build it first with scripts/build_xpair_cache.py.
        if [ ! -d "$XPAIR_CACHE" ]; then
            echo "ERROR: traj cache '$XPAIR_CACHE' not found — build it first:" >&2
            echo "       python scripts/build_xpair_cache.py --out $XPAIR_CACHE ..." >&2
            exit 1
        fi
        MODE_ARGS+=(
            --policy.type=traj_hyper_lora_smolvla
            --policy.hn_use_traj_clip=true
            --policy.hn_xpair_cache_path="$XPAIR_CACHE"
            --policy.hn_traj_encoder="$ENC"
            --policy.hn_pair_mode="$PAIR"
            --policy.hn_context_k="$K"
            --policy.hn_inject_vlm_kv="$KV_FLAG"
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        ) ;;
esac

echo "==> Train | mode=$MODE | rank=$RANK | prec=$PREC | batch=$BATCH | seed=$SEED | aug=$AUG_FLAG | expert=$EXPERT_FLAG | wandb=$WANDB_FLAG"
echo "    output=$OUTPUT"
python train_hyper_lora.py \
    "${MODE_ARGS[@]}" \
    --dataset.repo_id=lerobot/libero \
    --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable="$AUG_FLAG" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers="$WORKERS" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --seed="$SEED" \
    --wandb.enable="$WANDB_FLAG" \
    --wandb.project="$WANDB_PROJECT" \
    --output_dir="$OUTPUT"

echo
echo "Done. Checkpoint: $OUTPUT/checkpoints/last/pretrained_model"
echo "Eval it with:  POLICIES=\"$MODE=$OUTPUT/checkpoints/last/pretrained_model\" bash scripts/eval.sh"
