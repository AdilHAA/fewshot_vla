#!/usr/bin/env bash
# Build every trajectory cache the token-ablation campaign needs, on 4x H100.
#
#   bash scripts/build_all_caches.sh                 # all three passes
#   PASS=dino  bash scripts/build_all_caches.sh      # just one
#   GPUS=8 bash scripts/build_all_caches.sh          # wider node
#
# THREE passes, not five caches. A build spends almost all of its wall clock on
# per-frame video decode, not on the encoder, so the plan minimises DECODE passes:
#
#   pass 1  dinov2-base            -> cls+patches  (+ derived cls)
#   pass 2  dinov2-with-registers  -> cls+reg4    (+ derived cls)
#   pass 3  vjepa2-vitl            -> tubelet_grid2
#
# `cls` is token 0 of every unit of the richer format, so the derived caches are
# BITWISE identical to a native build and cost neither GPU nor decode. Five caches
# for the price of three decode passes.
#
# Each pass is sharded across all GPUs; shards are balanced by FRAME COUNT (episodes
# run 75..505 frames, so round-robin would leave one GPU grinding alone), then merged
# on CPU into a cache byte-identical to an unsharded build.
#
# Env:
#   GPUS      (4)      number of GPUs to shard across
#   WORKERS   (12)     decode workers PER shard (decode is the bottleneck)
#   EB        (256)    dino frames per forward (throughput only, not semantic)
#   OUT       (outputs/xpair_cache)
#   PASS      (all)    all | bank | dino | dinoreg | vjepa
#   BANK      (outputs/frame_bank.npz)  t=0 frame bank; every arm needs it once
#                    VLM=1 is on, because the VLM stream's train frame must be a
#                    t=0 frame (matching eval), not the current step's.
#   KEEP_SHARDS (0)    1 = keep the per-shard dirs after a successful merge

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

GPUS="${GPUS:-4}"
WORKERS="${WORKERS:-12}"
EB="${EB:-256}"
OUT="${OUT:-outputs/xpair_cache}"
PASS="${PASS:-all}"
KEEP_SHARDS="${KEEP_SHARDS:-0}"
BANK="${BANK:-outputs/frame_bank.npz}"
BUILD=scripts/build_xpair_cache.py

# run_pass <primary_out> <extra build args...>
# Fans the shards across GPUs, waits, merges, then verifies the merged record count.
run_pass() {
    local out="$1"; shift
    local -a derived=()
    for a in "$@"; do
        case "$a" in *:*) [ "${a%%:*}" != "--also_emit" ] || true ;; esac
    done
    # collect --also_emit targets so they get merged too
    local prev="" a
    for a in "$@"; do
        [ "$prev" = "--also_emit" ] && derived+=("${a#*:}")
        prev="$a"
    done

    echo "==> pass -> $out  (${GPUS} shards x ${WORKERS} decode workers)"
    local pids=() i
    for ((i = 0; i < GPUS; i++)); do
        CUDA_VISIBLE_DEVICES="$i" python "$BUILD" --out "$out" \
            --shard "$i" --num_shards "$GPUS" --workers "$WORKERS" "$@" &
        pids+=($!)
    done
    local rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" -eq 0 ] || { echo "ERROR: a shard failed for $out — not merging." >&2; exit 1; }

    for d in "$out" "${derived[@]}"; do
        echo "--> merging $d"
        python "$BUILD" --out "$d" --merge_shards "$GPUS"
        python - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d, "index.json")))
h, recs = m["header"], m["records"]
size = os.path.getsize(os.path.join(d, "tokens.mmap"))
want = sum(r["length"] for r in recs) * int(h["d_enc"]) * 2
assert size == want, f"{d}: tokens.mmap {size} != expected {want}"
assert h["num_records"] == len(recs), f"{d}: header/record count mismatch"
print(f"    ok {d}: {len(recs)} records, {size/2**30:.2f} GB, "
      f"format={h['format']}, tokens_per_unit={h['tokens_per_unit']}, "
      f"encoder_model={h['encoder_model']}")
PY
        if [ "$KEEP_SHARDS" != "1" ]; then
            for ((i = 0; i < GPUS; i++)); do rm -rf "$d.shard$i"; done
        fi
    done
}

# --- pass 0: the t=0 frame bank (single-GPU, minutes: one frame per episode) -----
if [ "$PASS" = "all" ] || [ "$PASS" = "bank" ]; then
    if [ -f "$BANK" ]; then
        echo "==> frame bank already at $BANK — skipping"
    else
        echo "==> frame bank -> $BANK"
        mkdir -p "$(dirname "$BANK")"
        python scripts/build_frame_bank.py --out "$BANK"
    fi
fi

# --- pass 1: dinov2-base. Raw 257 tokens/frame + the cls cache for free ----------
if [ "$PASS" = "all" ] || [ "$PASS" = "dino" ]; then
    run_pass "$OUT/dino_raw" --encoder dino --dino_patches --encode_batch "$EB" \
        --also_emit "cls:$OUT/dino"
fi

# --- pass 2: dinov2-with-registers. 4 registers + its own cls control ------------
# A SEPARATE checkpoint, not a flag: cos(CLS_base, CLS_reg)=0.059. The control arm
# must read this cache, not the dinov2-base one — hence the encoder_model header.
if [ "$PASS" = "all" ] || [ "$PASS" = "dinoreg" ]; then
    run_pass "$OUT/dinoreg_cls_reg4" --encoder dino --dino_n_reg 4 \
        --encoder_model facebook/dinov2-with-registers-base --encode_batch "$EB" \
        --also_emit "cls:$OUT/dinoreg_cls"
fi

# --- pass 3: vjepa2 -------------------------------------------------------------
# `--chunk` here IS semantic: it bounds the temporal attention horizon of every
# token in the window (unlike dino, whose frames are embedded independently).
if [ "$PASS" = "all" ] || [ "$PASS" = "vjepa" ]; then
    run_pass "$OUT/vjepa2" --encoder vjepa2 --vjepa_grid 2 --chunk 32
fi

echo
echo "Done. Caches in $OUT:"
du -sh "$OUT"/* 2>/dev/null || true
