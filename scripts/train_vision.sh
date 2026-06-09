#!/usr/bin/env bash
# Train the vision-conditioned hypernetwork on ALL of lerobot/libero (40 tasks,
# 1693 episodes), conditioned on text + the frozen VLM's own image tokens + an
# external frozen DINOv2, then eval on a LIBERO-Pro suite. The SmolVLA base +
# DINO are frozen; only the hypernet (incl. its vision projections) trains.
# wandb on by default.
#
#   bash scripts/train_vision.sh
#
# Env vars (override as needed):
#   STEPS      (100000)                training steps
#   SEED       (42)                    training seed; non-default seeds get a
#                                      suffixed OUTPUT dir automatically
#   AUG        (0)                     1 = enable dataset image augmentations
#                                      (color jitter etc.) during training
#   EXPERT     (0)                     1 = also unfreeze + train the action
#                                      expert (checkpoint grows)
#   BATCH      (16)                    train batch size (vision+DINO are heavier
#                                      than text-only; lower if VRAM-bound)
#   WORKERS    (8)                     dataloader workers
#   SAVE_FREQ  (25000)                 checkpoint interval (steps)
#   OUTPUT     (outputs/hyper_lora_vision)  output dir (must not pre-exist)
#   PREC       (bf16)                  bf16 | no (fp32) | fp16(NaN — avoid)
#   DINO_ID    (facebook/dinov2-base)  external vision encoder (…-small = lighter)
#   WANDB      (1)                     1 = enable wandb logging
#   WANDB_PROJECT (hyper-lora-vision)  wandb project name
#   EVAL       (1)                     1 = run an eval after training
#   EVAL_TASK  (libero_object_swap)    LIBERO-Pro suite to eval on
#   EVAL_EPISODES (50)                 episodes per task
#   EVAL_BATCH (2)                     parallel envs per task (24GB → 2; see run_eval.sh)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
AUG="${AUG:-0}"
EXPERT="${EXPERT:-0}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
# Non-default seeds / ablation toggles get distinct default output dirs so
# multi-seed and ablation runs don't collide.
DEFAULT_OUTPUT="outputs/hyper_lora_vision"
[ "$SEED" != "42" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_s${SEED}"
[ "$AUG" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_aug"
[ "$EXPERT" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_expert"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
PREC="${PREC:-bf16}"
DINO_ID="${DINO_ID:-facebook/dinov2-base}"
export ACCELERATE_MIXED_PRECISION="$PREC"
[ "${WANDB:-1}" = "1" ] && WANDB_FLAG=true || WANDB_FLAG=false
[ "$AUG" = "1" ] && AUG_FLAG=true || AUG_FLAG=false
[ "$EXPERT" = "1" ] && EXPERT_FLAG=true || EXPERT_FLAG=false
WANDB_PROJECT="${WANDB_PROJECT:-hyper-lora-vision}"

EVAL="${EVAL:-1}"
EVAL_TASK="${EVAL_TASK:-libero_object_swap}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_BATCH="${EVAL_BATCH:-2}"

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output dir '$OUTPUT' already exists (lerobot refuses to overwrite)."
    echo "       Set OUTPUT=… to a fresh path or remove it."
    exit 1
fi

# Work around lerobot/libero's wrong meta/episodes file_index (idempotent, ~21MB):
# pre-fetch every data parquet so the loader globs + filters by episode_index.
python -c "from src.data.libero import prefetch_all_data_parquets as p; p()"

echo "==> Train (vision) | vlm_vision=ON dino=ON ($DINO_ID) | prec=$PREC | batch=$BATCH | seed=$SEED | aug=$AUG_FLAG | expert=$EXPERT_FLAG | wandb=$WANDB_FLAG"
python train_hyper_lora.py \
    --policy.type=hyper_lora_smolvla \
    --policy.hn_use_vlm_vision=true \
    --policy.hn_use_dino=true \
    --policy.hn_dino_model_id="$DINO_ID" \
    --policy.train_action_expert="$EXPERT_FLAG" \
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

CKPT="$OUTPUT/checkpoints/last/pretrained_model"
echo
echo "Done. Checkpoint: $CKPT"

if [ "$EVAL" = "1" ]; then
    # Headless EGL render for the sim eval; pin the GPU device.
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
    echo "==> Eval on $EVAL_TASK | episodes=$EVAL_EPISODES | batch=$EVAL_BATCH (total envs=$((EVAL_BATCH * 10)))"
    python eval_hyper_lora.py \
        --policy.path="$CKPT" \
        --env.type=libero --env.task="$EVAL_TASK" \
        --eval.n_episodes="$EVAL_EPISODES" --eval.batch_size="$EVAL_BATCH" \
        --policy.device=cuda --policy.use_amp=false
else
    echo "Eval skipped (EVAL=0). Run it later with:"
    echo "  CKPT=$CKPT TASK=$EVAL_TASK EPISODES=$EVAL_EPISODES BATCH=$EVAL_BATCH bash scripts/run_eval.sh"
fi
