"""Rename feature keys of a lerobot v3.0 dataset IN PLACE (T2 step 3).

lerobot 0.5.1 ships delete/split/merge/remove_feature operations but NO rename,
and the converted yzembodied LIBERO-90 uses `observation.images.wrist_image`
where smolvla_libero (and lerobot/libero, and our whole eval env mapping) uses
`observation.images.image2`. Renaming the DATASET once is cleaner than carrying
--rename_map through every train/eval command forever.

Touches every place a feature key appears in the v3 layout:
  meta/info.json          feature dict keys (string-replace inside the JSON)
  meta/stats.json         stats dict keys
  meta/episodes/**/*.parquet   per-episode stats columns (e.g. "stats/<key>/mean")
  data/**/*.parquet       the data columns themselves
  videos/<key>/           directory name (video-dtype datasets only)

  python scripts/rename_libero90_features.py --root outputs/libero90/libero_90_image \
      --map observation.images.wrist_image=observation.images.image2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rename_json(path: Path, mapping: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text()
    new = text
    for old, neu in mapping.items():
        new = new.replace(f'"{old}"', f'"{neu}"').replace(f'"{old}/', f'"{neu}/')
    if new != text:
        json.loads(new)                       # sanity: still valid JSON
        path.write_text(new)
        return True
    return False


def rename_parquet(path: Path, mapping: dict[str, str]) -> bool:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = list(table.schema.names)
    new_names = []
    for n in names:
        for old, neu in mapping.items():
            if n == old or n.startswith(old + "/") or f"/{old}/" in n or n.endswith("/" + old):
                n = n.replace(old, neu)
                break
        new_names.append(n)
    if new_names == names:
        return False
    table = table.rename_columns(new_names)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--map", nargs="+",
                   default=["observation.images.wrist_image=observation.images.image2"],
                   help="old=new pairs")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"{root} is not a lerobot dataset root (no meta/info.json)")
    mapping = dict(pair.split("=", 1) for pair in args.map)

    n = 0
    for f in ("info.json", "stats.json"):
        if rename_json(root / "meta" / f, mapping):
            print(f"renamed keys in meta/{f}")
            n += 1
    for pattern in ("meta/episodes/**/*.parquet", "data/**/*.parquet"):
        for pf in sorted(root.glob(pattern)):
            if rename_parquet(pf, mapping):
                n += 1
    print(f"rewrote {n} files")
    for old, neu in mapping.items():
        vdir = root / "videos" / old
        if vdir.is_dir():
            vdir.rename(root / "videos" / neu)
            print(f"renamed videos/{old} -> videos/{neu}")

    feats = json.loads((root / "meta" / "info.json").read_text())["features"]
    print("final feature keys:", sorted(feats))
    leftovers = [k for k in feats for old in mapping if old in k]
    if leftovers:
        raise SystemExit(f"rename incomplete: {leftovers}")


if __name__ == "__main__":
    main()
