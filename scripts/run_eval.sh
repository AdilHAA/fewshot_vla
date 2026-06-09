#!/usr/bin/env bash
# Eval the Hyper-LoRA SmolVLA policy on a LIBERO / LIBERO-Pro suite.
#
#   bash scripts/run_eval.sh
#   TASK=libero_object_swap BATCH=2 bash scripts/run_eval.sh
#
# Env vars (override as needed):
#   CKPT     (outputs/hyper_lora_full/checkpoints/last/pretrained_model)  policy path
#   TASK     (libero_object_swap)  LIBERO/LIBERO-Pro suite
#   EPISODES (10)                  episodes per task
#   BATCH    (2)                   parallel envs PER task = batch_size
#
# Memory note (24 GB GPU): every LIBERO env opens an EGL render context on the
# GPU (~0.46 GB each). Eval builds envs for ALL 10 tasks of a suite at once, so
# total envs = BATCH * 10 and they all coexist. This hits two limits:
#   BATCH=10 -> 100 envs -> system RAM OOM during build ("Killed")
#   BATCH=5  ->  50 envs -> ~23.5 GB VRAM, no room for the policy -> CUDA OOM
#   BATCH=2  ->  20 envs -> ~9 GB VRAM, ~37 GB RAM            -> fits, recommended
# BATCH must divide EPISODES (2/5/10) or the last wave's data is discarded.
# Run only ONE eval at a time — two processes share the GPU and OOM each other.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

CKPT="${CKPT:-outputs/hyper_lora_full/checkpoints/last/pretrained_model}"
TASK="${TASK:-libero_object_swap}"
EPISODES="${EPISODES:-10}"
BATCH="${BATCH:-2}"

# Headless EGL render: pin the GPU device (plain egl may pick a non-renderable one).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# Guard: refuse to start if something is already holding the GPU (avoids the
# two-process mutual-OOM that looks like "Process NNN has 23.53 GiB in use").
if command -v nvidia-smi >/dev/null 2>&1; then
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  if [ "${used:-0}" -gt 1000 ]; then
    echo "WARNING: GPU already has ${used} MiB in use — another process may be running."
    echo "         Eval may OOM. Free the GPU (nvidia-smi) or Ctrl-C now." >&2
    sleep 5
  fi
fi

echo "Eval | task=${TASK} | episodes=${EPISODES} | batch(envs/task)=${BATCH} | total envs=$((BATCH * 10))"
echo "ckpt=${CKPT}"

exec python eval_hyper_lora.py \
    --policy.path="${CKPT}" \
    --env.type=libero --env.task="${TASK}" \
    --eval.n_episodes="${EPISODES}" --eval.batch_size="${BATCH}" \
    --policy.device=cuda --policy.use_amp=false
