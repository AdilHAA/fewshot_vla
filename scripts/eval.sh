#!/usr/bin/env bash
# Eval: policies × suites × seeds, with a summary table at the end. A single
# policy/suite/seed is just a 1×1×1 matrix.
# Resumable — a cell whose eval_info.json already exists is skipped, so the
# script can be re-run after an interruption (or extended with more
# seeds/suites) and only computes what's missing.
#
#   bash scripts/eval.sh
#   POLICIES="lora=outputs/lora_baseline/checkpoints/last/pretrained_model" bash scripts/eval.sh
#   TASKS="libero_spatial libero_spatial_object" SEEDS="1000" bash scripts/eval.sh
#   TASKS="libero_90_train libero_90_eval" SEEDS="1000" bash scripts/eval.sh
#
# Env vars:
#   POLICIES  (base + vision)     space-separated label=policy.path pairs
#   TASKS     (object ID + 4 axes) space-separated suite names. Two PSEUDO-suites
#                                 exist beyond lerobot's own: libero_90_train and
#                                 libero_90_eval = --env.task=libero_90 restricted
#                                 to the task_ids of configs/libero90_split.json,
#                                 run in CHUNKS of L90_CHUNK tasks (every selected
#                                 task's MuJoCo env is created eagerly and lives
#                                 until the run ends — 50 at once would blow RAM).
#                                 Cells land in <label>/<pseudo>/chunk_<k>/seed_<s>.
#   SEEDS     (1000 2000 3000)    eval seeds (≥3 for stable numbers)
#   EPISODES  (50)                episodes per task (BATCH must divide it)
#   BATCH     (2)                 parallel envs per task (24GB -> 2)
#   OUT_ROOT  (outputs/eval_matrix)  root dir for all cells
#   SPLIT_FILE (configs/libero90_split.json)  the fixed 40/50 split
#   L90_CHUNK (10)                tasks per pseudo-suite chunk
#   L90_CHUNKS ()                 run only these chunk indices of a pseudo-suite
#                                 (space-separated, e.g. "0 2"); empty = all. Lets
#                                 a multi-GPU launcher spread the 5 libero_90_eval
#                                 chunks over cards — cells keep the chunk_<k> layout.
#   EPISODE_CACHE (1)                build the adapter once per episode and freeze it
#                                    (the v2 protocol for every conditioned arm);
#                                    set 0 to regenerate per inference (legacy)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

# Headless EGL render; pin the GPU device.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

POLICIES="${POLICIES:-base=HuggingFaceVLA/smolvla_libero vision=outputs/hyper_lora_vision/checkpoints/last/pretrained_model}"
TASKS="${TASKS:-libero_object libero_object_lan libero_object_object libero_object_swap libero_object_task}"
SEEDS="${SEEDS:-1000 2000 3000}"
EPISODES="${EPISODES:-50}"
BATCH="${BATCH:-2}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_matrix}"
SPLIT_FILE="${SPLIT_FILE:-configs/libero90_split.json}"
L90_CHUNK="${L90_CHUNK:-10}"
L90_CHUNKS="${L90_CHUNKS:-}"
# TASK_IDS: evaluate only these task ids WITHIN each suite (single-task check, e.g.
# the overfit diagnostic). Format is a python list: TASK_IDS="[3]". NO quotes inside.
# A 1-task cell would permanently shadow the full-suite cell of the same
# label/suite/seed (the skip-check below keys on eval_info.json existing), so it is
# REFUSED in the default OUT_ROOT — point OUT_ROOT at a separate tree.
TASK_IDS="${TASK_IDS:-}"
if [ -n "$TASK_IDS" ] && [ "$OUT_ROOT" = "outputs/eval_matrix" ]; then
    echo "ERROR: TASK_IDS=$TASK_IDS with the default OUT_ROOT would write a 1-task cell" >&2
    echo "       into outputs/eval_matrix/<label>/<suite>/seed_N — which then permanently" >&2
    echo "       shadows that label's future FULL-suite cell (the skip-check only tests" >&2
    echo "       for eval_info.json). Use a separate tree, e.g.:" >&2
    echo "       OUT_ROOT=outputs/eval_matrix_1task TASK_IDS=\"$TASK_IDS\" bash scripts/eval.sh" >&2
    exit 1
fi
[ "${EPISODE_CACHE:-1}" != "0" ] && export HN_LORA_CACHE="${HN_LORA_CACHE:-episode}"

# Guard: warn if the GPU this worker will use is already busy (two-process OOM).
# Query ONLY that card and read the whole output: `| head -1` on a multi-GPU
# node closed the pipe early, nvidia-smi died of SIGPIPE, and with pipefail+set -e
# the script exited silently before printing anything (all 8 workers "failed"
# with empty logs on the 8xA100 node).
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_id="${CUDA_VISIBLE_DEVICES%%,*}"; gpu_id="${gpu_id:-0}"
    used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n 1p || true)"
    if [ "${used:-0}" -gt 1000 ] 2>/dev/null; then
        echo "WARNING: GPU $gpu_id already has ${used} MiB in use — free it or Ctrl-C." >&2
        sleep 5
    fi
fi

# One eval cell. $1 label $2 policy path $3 display suite $4 env task
# $5 cell dir $6 task_ids python list ('' = all tasks) $7 seed
run_cell() {
    local label="$1" path="$2" tlabel="$3" etask="$4" cell="$5" ids="$6" seed="$7"
    if [ -f "$cell/eval_info.json" ]; then
        echo "==> skip [$label | $tlabel | ${cell#"$OUT_ROOT/$label/"}] — already done"
        return 0
    fi
    # An existing dir without eval_info.json is a crashed cell; redo it.
    rm -rf "$cell"
    echo "==> eval [$label | $tlabel | ${cell#"$OUT_ROOT/$label/"}] | episodes=$EPISODES | batch=$BATCH${ids:+ | task_ids=$ids}"
    # shellcheck disable=SC2086
    python eval_hyper_lora.py \
        --policy.path="$path" \
        --env.type=libero --env.task="$etask" \
        ${ids:+--env.task_ids=$ids} \
        --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" \
        --seed="$seed" \
        --policy.device=cuda --policy.use_amp=false \
        --output_dir="$cell"
}

for entry in $POLICIES; do
    label="${entry%%=*}"
    path="${entry#*=}"
    if [ "$path" != "${path#outputs/}" ] \
        && [ ! -f "$path/model.safetensors" ] && [ ! -f "$path/adapter_model.safetensors" ]; then
        echo "ERROR: [$label] checkpoint not found at $path" >&2
        exit 1
    fi
    for task in $TASKS; do
        case "$task" in
            libero_90_train|libero_90_eval)
                if [ -n "$TASK_IDS" ]; then
                    echo "ERROR: TASK_IDS cannot combine with pseudo-suite $task (its" >&2
                    echo "       task_ids come from $SPLIT_FILE). Use --env.task=libero_90." >&2
                    exit 1
                fi
                if [ ! -f "$SPLIT_FILE" ]; then
                    echo "ERROR: $task needs the split file '$SPLIT_FILE' (make_libero90_split.py)." >&2
                    exit 1
                fi
                part="${task#libero_90_}"
                ci=0
                while IFS= read -r ids; do
                    if [ -z "$L90_CHUNKS" ] || grep -qw "$ci" <<< "$L90_CHUNKS"; then
                        for seed in $SEEDS; do
                            run_cell "$label" "$path" "$task" libero_90 \
                                "$OUT_ROOT/$label/$task/chunk_$ci/seed_$seed" "$ids" "$seed"
                        done
                    fi
                    ci=$((ci + 1))
                done < <(python scripts/libero90_episodes.py --split "$SPLIT_FILE" \
                             --part "$part" --emit chunks --chunk "$L90_CHUNK")
                ;;
            *)
                for seed in $SEEDS; do
                    run_cell "$label" "$path" "$task" "$task" \
                        "$OUT_ROOT/$label/$task/seed_$seed" "$TASK_IDS" "$seed"
                done
                ;;
        esac
    done
done

echo
echo "==> Summary (pc_success, mean ± std over seeds)"
python scripts/summarize_matrix.py "$OUT_ROOT"
