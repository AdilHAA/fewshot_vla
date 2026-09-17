#!/usr/bin/env bash
# ARGS for the base finetune (SmolVLA paper recipe). Source it, then launch — see finetune_ours.sh.
# Knobs: BASE_PATH DATA_ROOT DATA_REPO EPISODES_FILE VIDEO_BACKEND WORKERS
#   defaults = Kesvill/libero_90_lerobot_v3 downloaded to outputs/libero90/libero_90_lerobot_v3,
#   train part = its train_episodes.json
BASE_PATH="${BASE_PATH:-outputs/base/smolvla_base_libero_io}"
DATA_REPO="${DATA_REPO:-Kesvill/libero_90_lerobot_v3}"
DATA_ROOT="${DATA_ROOT:-outputs/libero90/libero_90_lerobot_v3}"
EPISODES_FILE="${EPISODES_FILE:-$DATA_ROOT/train_episodes.json}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
WORKERS="${WORKERS:-12}"
case "$DATA_ROOT" in /*) ;; *) DATA_ROOT="$PWD/$DATA_ROOT" ;; esac

PY=python; command -v python >/dev/null 2>&1 || PY=python3
if [ ! -f "$EPISODES_FILE" ]; then
    echo "ERROR: $EPISODES_FILE not found — hf download $DATA_REPO --repo-type dataset --local-dir $DATA_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi
EPIS="$("$PY" -c 'import json,sys; print(json.dumps(sorted(set(json.load(open(sys.argv[1])))), separators=(",", ":")))' "$EPISODES_FILE")"

ARGS=(
    --policy.path="$BASE_PATH"
    --policy.push_to_hub=false
    --policy.device=cuda
    --policy.scheduler_decay_steps=100000
    --dataset.repo_id="$DATA_REPO"
    --dataset.root="$DATA_ROOT"
    --dataset.use_imagenet_stats=false
    --dataset.episodes="$EPIS"
    --dataset.video_backend="$VIDEO_BACKEND"
    --num_workers="$WORKERS"
    --save_freq=10000 --save_checkpoint=true --seed=42
    --wandb.enable=true
)
echo "ARGS ready: base=$BASE_PATH data=$DATA_ROOT episodes=$("$PY" -c "print(len($EPIS))") workers=$WORKERS"
