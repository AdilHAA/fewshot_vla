#!/usr/bin/env bash
# Eval on the LIBERO-Pro Object axis (OOD object appearance: recolored / bigger
# objects, e.g. red basket + yellow milk). Same scene & positions as base — this
# tests whether the policy visually recognizes the named object when its
# appearance is out-of-distribution.
#
# By default runs BOTH comparison points so you get the table in one shot:
#   base     = frozen SmolVLA                   (HuggingFaceVLA/smolvla_libero)
#   hypernet = trained hypernet checkpoint    (text + VLM-vision + DINO)
#
#   bash scripts/eval_object.sh
#   MODE=ckpt bash scripts/eval_object.sh            # only the hypernet checkpoint
#   MODE=base bash scripts/eval_object.sh            # only the base policy
#   TASK=libero_goal_object bash scripts/eval_object.sh
#
# Env vars (override as needed):
#   MODE      (both)                     both | base | ckpt
#   TASK      (libero_object_object)     Object-axis suite
#   EPISODES  (50)                       episodes per task (BATCH must divide it)
#   BATCH     (2)                        parallel envs per task (24GB -> 2; see run_eval.sh)
#   CKPT      (outputs/hyper_lora_vision/checkpoints/last/pretrained_model)  hypernet policy
#   BASE      (HuggingFaceVLA/smolvla_libero)                             base policy

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

# Headless EGL render: pin the GPU device (plain egl may pick a non-renderable one).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

MODE="${MODE:-both}"
TASK="${TASK:-libero_object_object}"
EPISODES="${EPISODES:-50}"
BATCH="${BATCH:-2}"
CKPT="${CKPT:-outputs/hyper_lora_vision/checkpoints/last/pretrained_model}"
BASE="${BASE:-HuggingFaceVLA/smolvla_libero}"

# Guard: refuse to start if the GPU is already busy (avoids two-process OOM).
if command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "${used:-0}" -gt 1000 ]; then
        echo "WARNING: GPU already has ${used} MiB in use — free it or Ctrl-C." >&2
        sleep 5
    fi
fi

run_one () {  # $1 = label, $2 = policy.path
    echo
    echo "==> [$1] eval on $TASK | episodes=$EPISODES | batch=$BATCH (total envs=$((BATCH * 10)))"
    echo "    policy=$2"
    python eval_hyper_lora.py \
        --policy.path="$2" \
        --env.type=libero --env.task="$TASK" \
        --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" \
        --policy.device=cuda --policy.use_amp=false
}

if [ "$MODE" = "base" ] || [ "$MODE" = "both" ]; then
    run_one "base" "$BASE"
fi
if [ "$MODE" = "ckpt" ] || [ "$MODE" = "both" ]; then
    if [ ! -f "$CKPT/model.safetensors" ]; then
        echo "ERROR: checkpoint not found at $CKPT" >&2
        exit 1
    fi
    run_one "hypernet" "$CKPT"
fi

echo
echo "Done. Success rate (pc_success) per run is in its eval_info.json under outputs/eval/<date>/."