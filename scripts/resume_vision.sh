#!/usr/bin/env bash
# Resume a training run from its last checkpoint (training_state: step/optimizer/
# scheduler/rng). lerobot resume needs --config_path=<.../train_config.json>
# + --resume=true; the config travels in the checkpoint.
#
#   bash scripts/resume_vision.sh
#
# Env vars:
#   RUN        (outputs/hyper_lora_vision)  run dir to resume
#   SAVE_FREQ  (25000)                      override saved save_freq

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

# Headless EGL render for any in-loop eval; pin the GPU device.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

RUN="${RUN:-outputs/hyper_lora_vision}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
CONFIG="$RUN/checkpoints/last/pretrained_model/train_config.json"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG not found — no checkpoint to resume from." >&2
    exit 1
fi

echo "==> Resuming $RUN from $(readlink -f "$RUN/checkpoints/last") | save_freq=$SAVE_FREQ"
python train_hyper_lora.py \
    --config_path="$CONFIG" \
    --resume=true \
    --save_freq="$SAVE_FREQ"
