#!/usr/bin/env bash
# MT-LoRA baseline — one static learnable LoRA (no hypernetwork, no
# conditioning) trained on ALL of lerobot/libero through the same injection
# sites / rank / alpha / steps as the hypernet runs. The control that separates
# "input-conditioning helps" from "any extra adaptation training helps".
#
#   bash scripts/train_mt_lora.sh
#   SEED=43 bash scripts/train_mt_lora.sh
#
# Env vars (override as needed):
#   STEPS      (100000)              training steps
#   SEED       (42)                  training seed; non-default seeds get a
#                                    suffixed OUTPUT dir automatically
#   AUG        (0)                   1 = enable dataset image augmentations
#   EXPERT     (0)                   1 = also unfreeze + train the action
#                                    expert (checkpoint grows)
#   RANK       (4)                   LoRA rank; 48 ≈ parameter parity with the
#                                    hypernet (parameter-parity baseline).
#                                    alpha is scaled to keep alpha/rank = 4
#   BATCH      (32)                  train batch size
#   WORKERS    (8)                   dataloader workers
#   SAVE_FREQ  (25000)               checkpoint interval (steps)
#   OUTPUT     (outputs/mt_lora)     output dir (must not pre-exist)
#   PREC       (bf16)                bf16 | no (fp32)
#   WANDB      (1)                   1 = enable wandb logging
#   WANDB_PROJECT (mt-lora)          wandb project name

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
AUG="${AUG:-0}"
EXPERT="${EXPERT:-0}"
RANK="${RANK:-4}"
ALPHA=$((RANK * 4))  # keep alpha/rank = 4 (the repo's default scaling)
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
DEFAULT_OUTPUT="outputs/mt_lora"
[ "$RANK" != "4" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_r${RANK}"
[ "$SEED" != "42" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_s${SEED}"
[ "$AUG" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_aug"
[ "$EXPERT" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_expert"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
PREC="${PREC:-bf16}"
export ACCELERATE_MIXED_PRECISION="$PREC"
[ "${WANDB:-1}" = "1" ] && WANDB_FLAG=true || WANDB_FLAG=false
[ "$AUG" = "1" ] && AUG_FLAG=true || AUG_FLAG=false
[ "$EXPERT" = "1" ] && EXPERT_FLAG=true || EXPERT_FLAG=false
WANDB_PROJECT="${WANDB_PROJECT:-mt-lora}"

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output dir '$OUTPUT' already exists (lerobot refuses to overwrite)."
    echo "       Set OUTPUT=… to a fresh path or remove it."
    exit 1
fi

# Work around lerobot/libero's wrong meta/episodes file_index (idempotent, ~21MB).
python -c "from src.data.libero import prefetch_all_data_parquets as p; p()"

echo "==> MT-LoRA train (static, unconditioned) | rank=$RANK alpha=$ALPHA | prec=$PREC | batch=$BATCH | seed=$SEED | aug=$AUG_FLAG | expert=$EXPERT_FLAG | wandb=$WANDB_FLAG"
python train_hyper_lora.py \
    --policy.type=hyper_lora_smolvla \
    --policy.static_lora=true \
    --policy.lora_rank="$RANK" \
    --policy.lora_alpha="$ALPHA" \
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

echo
echo "Done. Checkpoint: $OUTPUT/checkpoints/last/pretrained_model"
echo "Eval the matrix with:  POLICIES=\"mtlora=$OUTPUT/checkpoints/last/pretrained_model\" bash scripts/eval_matrix.sh"
