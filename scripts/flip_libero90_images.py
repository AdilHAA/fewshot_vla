"""Mirror image columns of a lerobot v3.0 image-parquet dataset left-right.

Why: the yzembodied LIBERO-90 conversion stores agentview frames horizontally
MIRRORED w.r.t. the simulator render lerobot's LIBERO env produces (raw LIBERO
frames are upside-down; OpenVLA/lerobot rotate them 180°, yzembodied evidently
only flipped vertically — the difference is a left-right mirror). A policy
trained on it reached 25-40% on its own training tasks; mirroring the env frames
at eval (EVAL_FLIP_LR) took the same checkpoint to 92.5% — so the fix is to put
the DATASET into the env convention once and retrain.

Rewrites every data/**/*.parquet: PNG bytes of the given image columns are
decoded, flipped (np.fliplr), re-encoded; everything else is copied verbatim
(schema + HF features metadata preserved). Writes a NEW root by default (meta/
copied; image stats are irrelevant for SmolVLA — VISUAL normalization is
IDENTITY). Parallel over files.

  python scripts/flip_libero90_images.py \
      --root outputs/libero90/libero_90_image \
      --out  outputs/libero90/libero_90_image_flipped \
      --keys observation.images.image            # add observation.images.image2 if the wrist is mirrored too
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _flip_png(b: bytes) -> bytes:
    im = Image.open(io.BytesIO(b))
    im.load()
    out = Image.fromarray(np.ascontiguousarray(np.fliplr(np.asarray(im))))
    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def flip_file(args):
    src, dst, keys = args
    table = pq.read_table(src)
    for k in keys:
        idx = table.schema.get_field_index(k)
        if idx < 0:
            raise KeyError(f"{src}: no column {k!r}; columns={table.schema.names}")
        field = table.schema.field(k)
        col = table.column(k).to_pylist()               # list of {"bytes":..., "path":...}
        new = [dict(d, bytes=_flip_png(d["bytes"])) if d and d.get("bytes") else d for d in col]
        table = table.set_column(idx, field, pa.array(new, type=field.type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return src.name, table.num_rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True, help="new dataset root (must not exist)")
    p.add_argument("--keys", nargs="+", default=["observation.images.image"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"{root} is not a lerobot dataset root")
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.mkdir(parents=True)
    shutil.copytree(root / "meta", out / "meta")          # info/stats/tasks/episodes: unchanged
    if (root / "videos").exists():
        raise SystemExit("video-backed dataset: this tool handles image (PNG-in-parquet) datasets only")

    files = sorted((root / "data").glob("*/*.parquet"))
    jobs = [(f, out / "data" / f.relative_to(root / "data"), args.keys) for f in files]
    print(f"flipping {args.keys} in {len(files)} files with {args.workers} workers -> {out}")
    n = 0
    with Pool(args.workers) as pool:
        for i, (name, rows) in enumerate(pool.imap_unordered(flip_file, jobs), 1):
            n += rows
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {n} frames", flush=True)
    # sanity: first row of the first file decodes and has the expected shape
    t = pq.read_table(jobs[0][1], columns=[args.keys[0]]).slice(0, 1)
    im = Image.open(io.BytesIO(t[args.keys[0]][0].as_py()["bytes"]))
    print(f"done: {n} frames; sample decoded {im.size} {im.mode}")


if __name__ == "__main__":
    main()
