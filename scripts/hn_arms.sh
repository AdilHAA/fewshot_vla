#!/usr/bin/env bash
# The two HN arms of the LIBERO-90 stage on the frozen Kesvill/smolvla_libero_90 base
# (plan §8): LoRA r16/a32 on the action expert's MLPs, cross pairing (context = another
# demo of the same task), trained on lerobot/libero + the 50 held-out LIBERO-90 tasks.
# A thin layer over scripts/train.sh: it only fixes the stage's knobs and prints them;
# every other train.sh variable passes through (STEPS, BATCH, SEED, RESUME, ...).
#
#   ARM=trunk   bash scripts/hn_arms.sh           # Qwen3.5-0.8B over every 4th frame, native video prompt
#   ARM=scratch bash scripts/hn_arms.sh           # 2-block transformer over DINOv2 CLS of all frames
#   ARM=trunk STEPS=300 OUTPUT=outputs/probe_trunk NPROC=1 GPUS=0 bash scripts/hn_arms.sh   # smoke
#
# Defaults size the run for a 4-GPU node: NPROC=4 x BATCH=8 = the global batch 32 of
# every HN arm so far. Two arms side by side: GPUS=0,1 NPROC=2 BATCH=16 PORT=29501.
#
# Data: bash scripts/prepare_hn_data.sh first (caches, episode keys, train list).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="${ARM:?set ARM=trunk|scratch}"
export MODE=traj PAIR=cross K=1 VLM=0 EXPERT=0 LORA_TARGET=expert_mlp
export RANK="${RANK:-16}" ALPHA="${ALPHA:-32}"
export BASE="${BASE:-Kesvill/smolvla_libero_90}"
export DATASET=local/libero_all DATASET_ROOT=outputs/libero90/libero_all VIDEO_BACKEND=pyav
export EPISODES_FILE="${EPISODES_FILE:-outputs/libero90/hn_train_episodes.json}"
export NPROC="${NPROC:-4}" BATCH="${BATCH:-8}" WORKERS="${WORKERS:-8}"
export STEPS="${STEPS:-100000}" SAVE_FREQ="${SAVE_FREQ:-10000}"
export SCHED_DECAY="${SCHED_DECAY:-$STEPS}"     # cosine over the whole run, not lerobot's 30k
export TB="${TB:-1}" WANDB="${WANDB:-0}"
export OUTPUT="${OUTPUT:-outputs/hn_$ARM}"
case "$ARM" in
    trunk)
        export ENC=dino TRUNK="${TRUNK:-Qwen/Qwen3.5-0.8B}" TSTRIDE="${TSTRIDE:-4}" TNATIVE="${TNATIVE:-1}" TEXT="${TEXT:-1}"
        export XPAIR_CACHE="${XPAIR_CACHE:-outputs/xpair_cache/hn_qwen35vl_e$TSTRIDE}" ;;
    scratch)
        export ENC=dino TENC_MODEL="${TENC_MODEL:-facebook/dinov2-base}"
        export XPAIR_CACHE="${XPAIR_CACHE:-outputs/xpair_cache/hn_dino}" ;;
    *) echo "ERROR: ARM must be trunk|scratch, got '$ARM'" >&2; exit 1 ;;
esac

echo "==> HN arm=$ARM | base=$BASE | cache=$XPAIR_CACHE | rank/alpha=$RANK/$ALPHA | nproc=$NPROC x batch=$BATCH | steps=$STEPS | output=$OUTPUT"
exec bash scripts/train.sh
