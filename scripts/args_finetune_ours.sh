#!/usr/bin/env bash
# ARGS for the base finetune (SmolVLA paper recipe). Source it, then launch — see finetune_ours.sh.
# Knobs: BASE_PATH DATA_ROOT DATA_REPO WORKERS
#        EPISODES_FILE  json list of dataset episodes (default: yzembodied train-40 = task_id*50+k)
#        VIDEO_BACKEND  e.g. pyav for video-backed datasets (NVIDIA AV1)
BASE_PATH="${BASE_PATH:-outputs/base/smolvla_base_libero_io}"
DATA_ROOT="${DATA_ROOT:-outputs/libero90/libero_90_image_flipped}"
DATA_REPO="${DATA_REPO:-yzembodied/libero_90_image}"
WORKERS="${WORKERS:-12}"
case "$DATA_ROOT" in /*) ;; *) DATA_ROOT="$PWD/$DATA_ROOT" ;; esac

PY=python; command -v python >/dev/null 2>&1 || PY=python3
if [ -n "${EPISODES_FILE:-}" ]; then
    EPIS="$("$PY" -c 'import json,sys; print(json.dumps(sorted(set(json.load(open(sys.argv[1])))), separators=(",", ":")))' "$EPISODES_FILE")"
else
    EPIS="$("$PY" scripts/libero90_episodes.py --part train 2>/dev/null)"
fi
if [ -z "$EPIS" ] || [ "$EPIS" = "[]" ]; then
    echo "ERROR: empty episode list (run from the repo root with the venv active)" >&2
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
[ -z "${VIDEO_BACKEND:-}" ] || ARGS+=(--dataset.video_backend="$VIDEO_BACKEND")
echo "ARGS ready: base=$BASE_PATH data=$DATA_ROOT episodes=$("$PY" -c "print(len($EPIS))") workers=$WORKERS"
