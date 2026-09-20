#!/usr/bin/env bash
# HN stage data prep on the GPU node, idempotent (every step is skipped when its
# output exists). Produces, in order:
#   outputs/libero90/libero_all                merged lerobot/libero + LIBERO-90 root
#   configs/task_registry.json                 verified against the installed LIBERO
#   outputs/libero90/libero_all/episode_keys.json   episode_index -> bddl stem
#   outputs/libero90/hn_train_episodes.json    1693 lerobot/libero + 2188 held-out eps
#   outputs/xpair_cache/hn_qwen35vl_e$EVERY    trunk arm cache (every EVERY-th frame)
#   outputs/xpair_cache/hn_dino                scratch arm cache (DINOv2 CLS, all frames)
# The caches cover ALL 5614 episodes / 130 tasks (the trainer restricts itself to
# hn_train_episodes.json): eval needs a demo of every task it runs, including the
# 39 90-train tasks of the "HN does no harm" control row. The hn_ prefix keeps them
# apart from the lerobot/libero-only caches of earlier stages (same header, 40 tasks).
#
#   bash scripts/prepare_hn_data.sh
#   GPUS=0,1 WORKERS=8 bash scripts/prepare_hn_data.sh
#
# Env vars:
#   GPUS     (0,1,2,3)          cards to shard the cache builds over
#   WORKERS  (12)               video-decode workers per shard
#   EVERY    (4)                trunk frame stride (= TSTRIDE at train time)
#   K90      (outputs/libero90/libero_90_lerobot_v3)  Kesvill root (episode_task_map,
#                               eval_episodes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

GPUS="${GPUS:-0,1,2,3}"
WORKERS="${WORKERS:-12}"
EVERY="${EVERY:-4}"
K90="${K90:-outputs/libero90/libero_90_lerobot_v3}"
ROOT=outputs/libero90/libero_all
KEYS=$ROOT/episode_keys.json
EPS=outputs/libero90/hn_train_episodes.json
OFFSET=1693   # lerobot/libero episodes come first in the merge; LIBERO-90 = Kesvill id + 1693
LOGS=outputs/cache_logs
mkdir -p "$LOGS"

[ -f "$K90/eval_episodes.json" ] || { echo "ERROR: $K90 missing (hf download Kesvill/libero_90_lerobot_v3 --repo-type dataset --local-dir $K90)" >&2; exit 1; }

echo "==> 1/6 merged root $ROOT"
[ -d "$ROOT" ] || python scripts/check_libero_merge.py --merge --out "$ROOT"

echo "==> 2/6 task registry check (LIBERO task order)"
python scripts/make_task_registry.py --check

echo "==> 3/6 episode keys $KEYS"
[ -f "$KEYS" ] || python scripts/make_episode_keys.py --root "$ROOT" \
    --libero90_map "$K90/episode_task_map.json" --libero90_offset $OFFSET --out "$KEYS"

echo "==> 4/6 HN train episodes $EPS"
[ -f "$EPS" ] || python -c "
import json, sys
k90, out, off = sys.argv[1:4]
ev = json.load(open(k90 + '/eval_episodes.json'))
eps = list(range(int(off))) + [e + int(off) for e in ev]
json.dump(eps, open(out, 'w'))
print(len(eps), 'episodes:', int(off), 'lerobot/libero +', len(ev), 'LIBERO-90 held-out')
" "$K90" "$EPS" $OFFSET

SRC=(--repo_id local/libero_all --root "$ROOT" --video_backend pyav --keys "$KEYS" --workers "$WORKERS")
IFS=, read -r -a CARDS <<< "$GPUS"
N=${#CARDS[@]}

# build <out> <builder.py> [builder flags]: one shard per card, then merge on CPU.
build() {
    local out=$1; shift
    for i in "${!CARDS[@]}"; do
        echo "    shard $i/$N on GPU ${CARDS[$i]} -> $LOGS/$(basename "$out").shard$i.log"
        CUDA_VISIBLE_DEVICES=${CARDS[$i]} python "$@" "${SRC[@]}" --out "$out" \
            --shard "$i" --num_shards "$N" > "$LOGS/$(basename "$out").shard$i.log" 2>&1 &
    done
    wait
    python "$@" --out "$out" --merge_shards "$N"
    rm -rf "$out".shard*
}

QWEN=outputs/xpair_cache/hn_qwen35vl_e$EVERY
DINO=outputs/xpair_cache/hn_dino
echo "==> 5/6 trunk cache $QWEN"
[ -d "$QWEN" ] || build "$QWEN" scripts/build_qwen35_video_cache.py --every "$EVERY" --model Qwen/Qwen3.5-0.8B

echo "==> 6/6 scratch cache $DINO"
[ -d "$DINO" ] || build "$DINO" \
    scripts/build_xpair_cache.py --encoder dino --encoder_model facebook/dinov2-base --encode_batch 256

python - "$QWEN" "$DINO" <<'PY'
import sys
from src.traj_data.traj_cache import TrajCache
for d in sys.argv[1:]:
    c = TrajCache(d)
    n_tasks, n_eps = len(c.task_keys()), len({r["episode"] for r in c.records})
    print(f"{d}: episodes={n_eps} tasks={n_tasks} d_enc={c.header['d_enc']} "
          f"format={c.header['format']} stride={c.header.get('stride')}")
    assert n_tasks == 130 and n_eps == 5614, (d, n_tasks, n_eps)
PY
