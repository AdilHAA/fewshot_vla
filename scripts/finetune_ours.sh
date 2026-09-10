#!/usr/bin/env bash
# Launch the smolvla_libero_ours finetune (probe or full) with ONE short command —
# long pasted lines get split by the cluster terminal. Prints the exact argv first.
#
#   NPROC=4 STEPS=300    OUT=outputs/ddp_probe_v2           LOCAL_CACHE=1 bash scripts/finetune_ours.sh
#   NPROC=4 STEPS=100000 OUT=outputs/smolvla_libero_ours_v2 LOCAL_CACHE=1 bash scripts/finetune_ours.sh
#   NPROC=1 STEPS=100000 OUT=outputs/smolvla_libero_ours_v2 GPUS=4 bash scripts/finetune_ours.sh
#
# Env knobs:
#   NPROC (1)        processes/GPUs; >1 = accelerate --multi_gpu (NCCL P2P/IB off — SHM works here)
#   GPUS  ()         CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3" or "4"; empty = all visible
#   STEPS (100000)   training steps
#   OUT   (outputs/smolvla_libero_ours_v2)   output dir (must not exist)
#   BATCH ()         per-process batch; default 64 for NPROC=1, 16 for NPROC=4 (global 64)
#   LOCAL_CACHE (0)  1 = read models/arrow cache/base ckpt from /tmp/local/{hub,datasets,base_io}
#                    (copies made with cp -r; NFS hung under concurrent multi-rank mmap reads)
#   + everything scripts/args_finetune_ours.sh understands (BASE_PATH, DATA_ROOT, WORKERS)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

NPROC="${NPROC:-1}"
STEPS="${STEPS:-100000}"
OUT="${OUT:-outputs/smolvla_libero_ours_v2}"
if [ -z "${BATCH:-}" ]; then [ "$NPROC" = "1" ] && BATCH=64 || BATCH=$((64 / NPROC)); fi
if [ -e "$OUT" ]; then echo "ERROR: $OUT exists — rm -rf it or pick another OUT" >&2; exit 1; fi

if [ "${LOCAL_CACHE:-0}" = "1" ]; then
    for d in hub datasets base_io; do
        [ -d "/tmp/local/$d" ] || { echo "ERROR: /tmp/local/$d missing (cp -r it first)" >&2; exit 1; }
    done
    export HF_HUB_CACHE=/tmp/local/hub HF_DATASETS_CACHE=/tmp/local/datasets
    export BASE_PATH="${BASE_PATH:-/tmp/local/base_io}"
fi
# shellcheck disable=SC1091
source scripts/args_finetune_ours.sh

export TENSORBOARD=1 DATASETS_SOFT_LOCK=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
LAUNCH=(accelerate launch --num_processes="$NPROC" --mixed_precision=bf16)
[ "$NPROC" != "1" ] && LAUNCH+=(--multi_gpu)
CMD=("${LAUNCH[@]}" train_hyper_lora.py "${ARGS[@]}" --batch_size="$BATCH" --steps="$STEPS" --output_dir="$OUT")

mkdir -p "$(dirname "$OUT")"
echo "==> nproc=$NPROC gpus=${GPUS:-all} batch/proc=$BATCH steps=$STEPS out=$OUT local_cache=${LOCAL_CACHE:-0}"
printf '    %s\n' "${CMD[@]}" | sed 's/episodes=\[.*\]/episodes=[...2000 ids...]/'
echo "==> log: $OUT.log"
"${CMD[@]}" 2>&1 | tee "$OUT.log"
