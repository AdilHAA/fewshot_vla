#!/usr/bin/env bash
# Spread the held-out-stage eval matrix of ONE policy over 8 GPUs.
#
# Work units (each ~10 tasks x EPISODES episodes, i.e. roughly equal wall time):
#   libero_90_eval  = 5 chunks of 10 held-out tasks   (chunk_0 .. chunk_4)
#   4 stock suites  = libero_10 libero_goal libero_object libero_spatial
#   4 LIBERO-Pro    = libero_10_lan libero_10_object libero_10_swap libero_10_task
# 13 units / 8 cards -> the busiest card gets 2 units, wall time ~ 2 units.
# Each card runs its units SEQUENTIALLY via scripts/eval.sh into ONE shared
# OUT_ROOT, so the matrix stays resumable (re-run this script after a crash:
# finished cells are skipped) and scripts/summarize_matrix.py reads it as usual.
#
#   bash scripts/eval_all_8gpu.sh ours=outputs/smolvla_libero_ours/checkpoints/last/pretrained_model
#
# Env vars: SEEDS (1000) EPISODES (50) BATCH (10) OUT_ROOT (outputs/eval_matrix)
#           GPUS ("0 1 2 3 4 5 6 7")  LOG_DIR (outputs/eval_logs)
#           ASSIGN  override the per-card table: rows "<TASKS>|<L90_CHUNKS>" joined
#                   by ';' — one row per GPU in GPUS. Example, the in-distribution
#                   check (40 train tasks = 4 chunks of 10) on four cards:
#             GPUS="0 1 2 3" ASSIGN="libero_90_train|0;libero_90_train|1;libero_90_train|2;libero_90_train|3" \
#                 bash scripts/eval_all_8gpu.sh ours=<ckpt>
#           PER_TASK (libero_90_eval)  suite whose per-task table the summary prints

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

POLICY="${1:?usage: $0 label=path/to/pretrained_model}"
SEEDS="${SEEDS:-1000}"
EPISODES="${EPISODES:-50}"
BATCH="${BATCH:-10}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_matrix}"
LOG_DIR="${LOG_DIR:-outputs/eval_logs}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"

# One row per card: "<TASKS>|<L90_CHUNKS>"  (L90_CHUNKS applies to the libero_90_*
# pseudo-suites only). Default = the held-out matrix over 8 cards; env ASSIGN
# (rows joined by ';') replaces the table.
if [ -n "${ASSIGN:-}" ]; then
    IFS=';' read -r -a ASSIGN <<< "$ASSIGN"
else
    ASSIGN=(
        "libero_90_eval libero_10|0"
        "libero_90_eval libero_goal|1"
        "libero_90_eval libero_object|2"
        "libero_90_eval libero_spatial|3"
        "libero_90_eval libero_10_lan|4"
        "libero_10_object|"
        "libero_10_swap|"
        "libero_10_task|"
    )
fi
PER_TASK="${PER_TASK:-libero_90_eval}"
if [ "${#GPUS[@]}" -ne "${#ASSIGN[@]}" ]; then
    echo "ERROR: ${#GPUS[@]} GPUs given but ASSIGN has ${#ASSIGN[@]} rows" >&2; exit 1
fi

mkdir -p "$LOG_DIR"
label="${POLICY%%=*}"
echo "==> policy=$POLICY | seeds=$SEEDS | episodes=$EPISODES | batch=$BATCH | out=$OUT_ROOT"
pids=()
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    tasks="${ASSIGN[$i]%%|*}"
    chunks="${ASSIGN[$i]#*|}"
    log="$LOG_DIR/${label}_gpu${gpu}.log"
    echo "    GPU $gpu: $tasks${chunks:+ (libero_90_eval chunks: $chunks)}  -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    POLICIES="$POLICY" TASKS="$tasks" L90_CHUNKS="$chunks" \
    SEEDS="$SEEDS" EPISODES="$EPISODES" BATCH="$BATCH" OUT_ROOT="$OUT_ROOT" \
        bash scripts/eval.sh > "$log" 2>&1 &
    pids+=($!)
done

echo "==> ${#pids[@]} workers running; follow with: tail -f $LOG_DIR/${label}_gpu*.log"
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "!! worker on GPU ${GPUS[$i]} FAILED — see $LOG_DIR/${label}_gpu${GPUS[$i]}.log" >&2
        fail=1
    fi
done

echo
echo "==> Summary"
python scripts/summarize_matrix.py "$OUT_ROOT" --per_task "$PER_TASK"
[ "$fail" = 0 ] || { echo "Some workers failed; re-run the same command — finished cells are skipped." >&2; exit 1; }
