#!/usr/bin/env bash
# Defines the ARGS array for the smolvla_libero_ours finetune (paper recipe:
# frozen VLM, expert only, lr 1e-4 cosine->2.5e-6 over the full horizon).
# Source it, then launch with a one-liner — multi-line pastes kept breaking in
# the cluster terminal, this file makes the argv reproducible instead.
#
#   source scripts/args_finetune_ours.sh              # defaults below
#   BASE_PATH=/tmp/local/base_io source scripts/args_finetune_ours.sh
#
# Env knobs (all optional):
#   BASE_PATH   (outputs/base/smolvla_base_libero_io)     LIBERO-I/O copy of smolvla_base
#   DATA_ROOT   (outputs/libero90/libero_90_image_flipped) converted+renamed+FLIPPED LIBERO-90
#   DATA_REPO   (yzembodied/libero_90_image)
#   WORKERS     (12)
# Not in ARGS on purpose (pass on the command line): --batch_size, --steps, --output_dir.
BASE_PATH="${BASE_PATH:-outputs/base/smolvla_base_libero_io}"
DATA_ROOT="${DATA_ROOT:-outputs/libero90/libero_90_image_flipped}"
DATA_REPO="${DATA_REPO:-yzembodied/libero_90_image}"
WORKERS="${WORKERS:-12}"
case "$DATA_ROOT" in /*) ;; *) DATA_ROOT="$PWD/$DATA_ROOT" ;; esac

PY=python; command -v python >/dev/null 2>&1 || PY=python3
EPIS="$("$PY" scripts/libero90_episodes.py --part train 2>/dev/null)"
if [ -z "$EPIS" ]; then
    echo "ERROR: could not build the train episode list (run from the repo root with the venv active)" >&2
    return 1 2>/dev/null || exit 1
fi
ARGS=(
    --policy.path="$BASE_PATH"
    --policy.push_to_hub=false
    --policy.device=cuda
    --policy.scheduler_decay_steps=100000
    --dataset.repo_id="$DATA_REPO"
    --dataset.root="$DATA_ROOT"
    --dataset.use_imagenet_stats=false
    --dataset.episodes="$EPIS"
    --num_workers="$WORKERS"
    --save_freq=10000 --save_checkpoint=true --seed=42
    --wandb.enable=true
)
echo "ARGS ready: base=$BASE_PATH data=$DATA_ROOT episodes=$("$PY" -c "print(len($EPIS))") workers=$WORKERS"
