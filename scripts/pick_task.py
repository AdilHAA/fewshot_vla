"""Task -> episode list + stats, for the single-task overfit arm.

Reads the embedding cache's index.json — every record already carries
{episode, variant, task_index} and the header says how many tokens one frame is,
so task tables and per-task frame counts come free, no dataset access needed.

  python scripts/pick_task.py outputs/xpair_cache/dino            # list all tasks
  python scripts/pick_task.py outputs/xpair_cache/dino 7          # task 7 detail
  python scripts/pick_task.py outputs/xpair_cache/dino 7 --shell  # ready EPISODES=
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def task_table(records: list, header: dict) -> dict:
    """{task_index: {"episodes": [...], "frames": n, "instruction": str}} for
    ORIGINAL recordings only (variant==0), matching the conditioning pool."""
    # tokens -> frames: dino 'cls' keeps 1 token per frame; vjepa tubelets cover
    # 2 frames and grid² tokens each. Derive from the format tag.
    fmt = header.get("format", "cls")
    if fmt.startswith("tubelet_grid"):
        g = int(fmt[len("tubelet_grid"):])
        per_frame = g * g / 2.0
    else:
        per_frame = float(header.get("tokens_per_unit", 1))

    table: dict = {}
    for r in records:
        if r.get("variant", 0) != 0:
            continue
        t = table.setdefault(int(r["task_index"]),
                             {"episodes": [], "frames": 0.0})
        t["episodes"].append(int(r["episode"]))
        t["frames"] += r["length"] / per_frame
    return table


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cache", help="cache dir with index.json")
    p.add_argument("task", nargs="?", type=int, help="task_index to detail")
    p.add_argument("--shell", action="store_true",
                   help="print only the ready-to-paste EPISODES= line")
    args = p.parse_args(argv)

    with open(os.path.join(args.cache, "index.json")) as fh:
        meta = json.load(fh)
    header, records = meta["header"], meta["records"]
    texts = {int(k): v for k, v in meta.get("task_texts", {}).items()}
    table = task_table(records, header)

    if args.task is None:
        print(f"{'task':>4} {'eps':>4} {'frames':>7} {'med_len':>8}  instruction")
        for t in sorted(table):
            e = table[t]
            print(f"{t:>4} {len(e['episodes']):>4} {int(e['frames']):>7} "
                  f"{'':>8}  {texts.get(t, '?')[:60]}")
        print("\ndetail + episode list: python scripts/pick_task.py "
              f"{args.cache} <task_index>")
        return

    if args.task not in table:
        sys.exit(ftask_err(table, args.task))
    e = table[args.task]
    eps = sorted(e["episodes"])
    lengths = sorted(r["length"] for r in records
                     if r.get("variant", 0) == 0 and r["task_index"] == args.task)
    if args.shell:
        print("EPISODES=\"[" + ",".join(map(str, eps)) + "]\"")
        return
    print(f"task {args.task}: {texts.get(args.task, '?')}")
    print(f"  episodes: {len(eps)}, frames: {int(e['frames'])}, "
          f"record lengths min/med/max: {lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}")
    print(f"  ~{e['frames']/32:.0f} steps/epoch at BATCH=32 "
          f"-> 30k steps = {30000/(e['frames']/32):.0f} epochs")
    print("\ntrain on just this task:")
    print("  EPISODES=\"[" + ",".join(map(str, eps)) + "]\" \\\n"
          "    STEPS=30000 MODE=traj ... bash scripts/train.sh")


def ftask_err(table, task):
    return (f"task {task} not in cache (have {sorted(table)[:5]}... "
            f"{sorted(table)[-1]}, {len(table)} tasks)")


if __name__ == "__main__":
    main()
