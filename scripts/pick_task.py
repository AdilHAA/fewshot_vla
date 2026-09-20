"""Task -> episode list + stats, for the single-task overfit arm.

Reads the embedding cache's index.json — every record already carries
{episode, variant, task_key} and the header says how many tokens one frame is, so
task tables and per-task frame counts come free, no dataset access needed.
Pre-registry caches carry an int `task_index` instead of the bddl stem; both load,
and the argument is whatever the table prints.

  python scripts/pick_task.py outputs/xpair_cache/dino          # list all tasks
  python scripts/pick_task.py outputs/xpair_cache/dino KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet
  python scripts/pick_task.py outputs/xpair_cache/dino 7 --shell  # ready EPISODES=
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def record_task(r: dict):
    """Task identity of a record: the bddl stem, or a legacy int task_index."""
    k = r.get("task_key")
    return k if k is not None else int(r["task_index"])


def task_table(records: list, header: dict) -> dict:
    """{task: {"episodes": [...], "frames": n}} for ORIGINAL recordings only
    (variant==0), matching the conditioning pool."""
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
        t = table.setdefault(record_task(r), {"episodes": [], "frames": 0.0})
        t["episodes"].append(int(r["episode"]))
        t["frames"] += r["length"] / per_frame
    return table


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cache", help="cache dir with index.json")
    p.add_argument("task", nargs="?", help="task_key (bddl stem) or legacy task_index")
    p.add_argument("--shell", action="store_true",
                   help="print only the ready-to-paste EPISODES= line")
    args = p.parse_args(argv)

    with open(os.path.join(args.cache, "index.json")) as fh:
        meta = json.load(fh)
    header, records = meta["header"], meta["records"]
    texts = meta.get("task_texts", {})                     # str keys after json
    table = task_table(records, header)

    if args.task is None:
        w = max([len(str(t)) for t in table] + [4])
        print(f"{'task':<{w}} {'eps':>4} {'frames':>7}  instruction")
        for t in sorted(table):
            e = table[t]
            print(f"{str(t):<{w}} {len(e['episodes']):>4} {int(e['frames']):>7}  "
                  f"{texts.get(str(t), '?')[:60]}")
        print("\ndetail + episode list: python scripts/pick_task.py "
              f"{args.cache} <task>")
        return

    task = args.task
    if task not in table and task.isdigit() and int(task) in table:
        task = int(task)                                   # legacy task_index cache
    if task not in table:
        sys.exit(ftask_err(table, args.task))
    e = table[task]
    eps = sorted(e["episodes"])
    lengths = sorted(r["length"] for r in records
                     if r.get("variant", 0) == 0 and record_task(r) == task)
    if args.shell:
        print("EPISODES=\"[" + ",".join(map(str, eps)) + "]\"")
        return
    print(f"task {task}: {texts.get(str(task), '?')}")
    print(f"  episodes: {len(eps)}, frames: {int(e['frames'])}, "
          f"record lengths min/med/max: {lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}")
    print(f"  ~{e['frames']/32:.0f} steps/epoch at BATCH=32 "
          f"-> 30k steps = {30000/(e['frames']/32):.0f} epochs")
    print("\ntrain on just this task:")
    print("  EPISODES=\"[" + ",".join(map(str, eps)) + "]\" \\\n"
          "    STEPS=30000 MODE=traj ... bash scripts/train.sh")


def ftask_err(table, task):
    keys = sorted(map(str, table))
    return (f"task {task!r} not in cache (have {keys[:3]}... {keys[-1]}, "
            f"{len(table)} tasks)")


if __name__ == "__main__":
    main()
