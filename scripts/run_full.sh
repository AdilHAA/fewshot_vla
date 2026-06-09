#!/usr/bin/env bash
# Full online Hyper-LoRA training run on all of lerobot/libero (40 tasks, 1693
# episodes). The SmolVLA base is frozen end-to-end; only the hypernet trains.
# Single GPU. Run from a venv built by scripts/setup.sh.
#
#   bash scripts/run_full.sh
#
# Env vars (override as needed):
#   STEPS      (100000)                 training steps
#   BATCH      (32)                     batch size (~7GB VRAM @16, fp32)
#   WORKERS    (8)                      dataloader workers
#   SAVE_FREQ  (10000)                  checkpoint interval (steps)
#   OUTPUT     (outputs/hyper_lora_full)  output dir (must not pre-exist)
#   PREC       (bf16)                    training precision: bf16 | no (fp32) | fp16
#   WANDB      (0)                       1 = enable wandb logging
#
# Precision note: lerobot-train's mixed precision is driven by Accelerate
# (ACCELERATE_MIXED_PRECISION), NOT by --policy.use_amp (that flag only affects
# eval). bf16 is fast on Ampere+ (4090/A100/H100) and numerically stable. Do
# NOT use fp16 — flow-matching loss goes NaN. On V100/older (no hardware bf16)
# set PREC=no for fp32 (bf16 there is stable but a bit slower).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

STEPS="${STEPS:-100000}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
OUTPUT="${OUTPUT:-outputs/hyper_lora_full}"
PREC="${PREC:-bf16}"
export ACCELERATE_MIXED_PRECISION="$PREC"
[ "${WANDB:-0}" = "1" ] && WANDB_FLAG=true || WANDB_FLAG=false

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output dir '$OUTPUT' already exists (lerobot refuses to overwrite)."
    echo "       Set OUTPUT=… to a fresh path or remove it."
    exit 1
fi

# Work around lerobot/libero's wrong meta/episodes file_index (idempotent, ~21MB):
# pre-fetch every data parquet so the loader globs + filters by episode_index.
python -c "from src.data.libero import prefetch_all_data_parquets as p; p()"

# Base SmolVLA config fields are auto-derived from the checkpoint by the shim.
echo "==> precision (ACCELERATE_MIXED_PRECISION) = $PREC"
python train_hyper_lora.py \
    --policy.type=hyper_lora_smolvla \
    --dataset.repo_id=lerobot/libero \
    --dataset.use_imagenet_stats=false \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers="$WORKERS" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --seed=42 \
    --wandb.enable="$WANDB_FLAG" \
    --output_dir="$OUTPUT"

echo
echo "Done. Checkpoints in $OUTPUT/checkpoints. Eval on LIBERO-Pro with:"
echo "  python eval_hyper_lora.py --policy.path=$OUTPUT/checkpoints/last/pretrained_model \\"
echo "    --env.type=libero --env.task=libero_object_swap \\"
echo "    --eval.n_episodes=10 --eval.batch_size=10 --policy.device=cuda --policy.use_amp=false"
