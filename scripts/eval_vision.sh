#!/usr/bin/env bash
# Eval the trained vision-conditioned checkpoint (text + VLM-vision + DINO) on a
# LIBERO-Pro suite. Base SmolVLA + DINO are reconstructed on load from the
# checkpoint config (base_smolvla_path / hn_dino_model_id); only the ~24MB
# hypernet is in the checkpoint.
#
#   bash scripts/eval_vision.sh                 # libero_object_swap
#   TASK=libero_object bash scripts/eval_vision.sh   # in-distribution sanity
#
# Env vars:
#   RUN       (outputs/hyper_lora_vision)  run dir
#   TASK      (libero_object_swap)         suite to eval
#   EPISODES  (50)                         episodes per task
#   BATCH     (2)                          parallel envs per task (24GB -> 2)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

# Headless EGL render; pin the GPU device.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

RUN="${RUN:-outputs/hyper_lora_vision}"
TASK="${TASK:-libero_object_swap}"
EPISODES="${EPISODES:-50}"
BATCH="${BATCH:-2}"
CKPT="$RUN/checkpoints/last/pretrained_model"

if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "ERROR: checkpoint not found at $CKPT" >&2
    exit 1
fi

# Guard: refuse to start if the GPU is already busy (avoids two-process OOM).
if command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "${used:-0}" -gt 1000 ]; then
        echo "WARNING: GPU already has ${used} MiB in use — free it or Ctrl-C." >&2
        sleep 5
    fi
fi

echo "==> Eval (vision) | task=$TASK | episodes=$EPISODES | batch=$BATCH (total envs=$((BATCH * 10)))"
echo "ckpt=$CKPT"
python eval_hyper_lora.py \
    --policy.path="$CKPT" \
    --env.type=libero --env.task="$TASK" \
    --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" \
    --policy.device=cuda --policy.use_amp=false
