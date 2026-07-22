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
#
# Env vars:
#   POLICIES  (base + vision)     space-separated label=policy.path pairs
#   TASKS     (object ID + 4 axes) space-separated suite names
#   SEEDS     (1000 2000 3000)    eval seeds (≥3 for stable numbers)
#   EPISODES  (50)                episodes per task (BATCH must divide it)
#   BATCH     (2)                 parallel envs per task (24GB -> 2)
#   OUT_ROOT  (outputs/eval_matrix)  root dir for all cells
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
[ "${EPISODE_CACHE:-1}" != "0" ] && export HN_LORA_CACHE="${HN_LORA_CACHE:-episode}"

# Guard: refuse to start if the GPU is already busy (avoids two-process OOM).
if command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "${used:-0}" -gt 1000 ]; then
        echo "WARNING: GPU already has ${used} MiB in use — free it or Ctrl-C." >&2
        sleep 5
    fi
fi

for entry in $POLICIES; do
    label="${entry%%=*}"
    path="${entry#*=}"
    if [ "$path" != "${path#outputs/}" ] \
        && [ ! -f "$path/model.safetensors" ] && [ ! -f "$path/adapter_model.safetensors" ]; then
        echo "ERROR: [$label] checkpoint not found at $path" >&2
        exit 1
    fi
    for task in $TASKS; do
        for seed in $SEEDS; do
            cell="$OUT_ROOT/$label/$task/seed_$seed"
            if [ -f "$cell/eval_info.json" ]; then
                echo "==> skip [$label | $task | seed=$seed] — already done"
                continue
            fi
            # An existing dir without eval_info.json is a crashed cell; redo it.
            rm -rf "$cell"
            echo "==> eval [$label | $task | seed=$seed] | episodes=$EPISODES | batch=$BATCH"
            python eval_hyper_lora.py \
                --policy.path="$path" \
                --env.type=libero --env.task="$task" \
                --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" \
                --seed="$seed" \
                --policy.device=cuda --policy.use_amp=false \
                --output_dir="$cell"
        done
    done
done

echo
echo "==> Summary (pc_success, mean ± std over seeds)"
python - "$OUT_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
cells = {}  # (label, task) -> [pc_success per seed]
for info in sorted(root.glob("*/*/seed_*/eval_info.json")):
    label, task = info.parts[-4], info.parts[-3]
    pc = json.load(open(info))["overall"]["pc_success"]
    cells.setdefault((label, task), []).append(pc)

if not cells:
    print("(no completed cells found)")
    sys.exit(0)

labels = sorted({k[0] for k in cells})
tasks = sorted({k[1] for k in cells})
w = max(len(t) for t in tasks) + 2
print("| policy | " + " | ".join(tasks) + " |")
print("|---" * (len(tasks) + 1) + "|")
for label in labels:
    row = [label]
    for task in tasks:
        v = cells.get((label, task))
        if not v:
            row.append("—")
        elif len(v) == 1:
            row.append(f"{v[0]:.1f}")
        else:
            row.append(f"{statistics.mean(v):.1f} ± {statistics.stdev(v):.1f}")
    print("| " + " | ".join(row) + " |")
PY
