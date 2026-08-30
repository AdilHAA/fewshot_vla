"""Aggregate an eval matrix — including chunked LIBERO-90 pseudo-suites — into a
markdown table (T4). Replaces the inline summary at the end of eval.sh.

Cell layouts understood:
  <root>/<label>/<suite>/seed_<S>/eval_info.json               (classic)
  <root>/<label>/<suite>/chunk_<K>/seed_<S>/eval_info.json     (libero_90_* chunks)

For chunked suites the per-seed number is the mean of per-task pc_success over
ALL chunks of that seed (equal episodes per task, so this equals the pooled
mean); task coverage is checked against configs/libero90_split.json when the
suite is libero_90_train/eval and the split file exists.

  python scripts/summarize_matrix.py outputs/eval_matrix
  python scripts/summarize_matrix.py outputs/eval_matrix --per_task libero_90_eval
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DEFAULT = os.path.join(ROOT, "configs/libero90_split.json")


def load_cells(root: Path):
    """-> {(label, suite, seed): {task_id: pc_success}}, {(label, suite, seed): overall_pc}"""
    per_task, overall = {}, {}
    for info_path in sorted(root.glob("*/*/seed_*/eval_info.json")) + sorted(
            root.glob("*/*/chunk_*/seed_*/eval_info.json")):
        parts = info_path.parts
        chunked = parts[-3].startswith("chunk_")
        label, suite = (parts[-5], parts[-4]) if chunked else (parts[-4], parts[-3])
        seed = parts[-2]
        info = json.load(open(info_path))
        key = (label, suite, seed)
        tasks = per_task.setdefault(key, {})
        for row in info.get("per_task", []):
            tid = int(row["task_id"])
            if tid in tasks:
                print(f"WARNING: duplicate task {tid} in {label}/{suite}/{seed}",
                      file=sys.stderr)
            tasks[tid] = float(row["metrics"]["pc_success"])
        if not chunked:
            overall[key] = float(info["overall"]["pc_success"])
    return per_task, overall


def seed_value(per_task, overall, key):
    """One number per (label, suite, seed): classic cells use the recorded
    overall; chunked cells use the mean over tasks."""
    if key in overall:
        return overall[key]
    tasks = per_task.get(key)
    return statistics.mean(tasks.values()) if tasks else None


def check_coverage(per_task, split_path: str):
    if not os.path.isfile(split_path):
        return
    split = json.load(open(split_path))
    want = {"libero_90_train": {t["task_id"] for t in split["train"]},
            "libero_90_eval": {t["task_id"] for t in split["eval"]}}
    for (label, suite, seed), tasks in per_task.items():
        if suite in want and set(tasks) != want[suite]:
            missing = sorted(want[suite] - set(tasks))
            extra = sorted(set(tasks) - want[suite])
            print(f"WARNING: {label}/{suite}/{seed} task coverage "
                  f"missing={missing} extra={extra}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--split", default=SPLIT_DEFAULT)
    p.add_argument("--per_task", default="",
                   help="also print the per-task table of this suite")
    args = p.parse_args(argv)

    root = Path(args.root)
    per_task, overall = load_cells(root)
    if not per_task and not overall:
        print("(no completed cells found)")
        return
    check_coverage(per_task, args.split)

    cells = {}  # (label, suite) -> [per-seed values]
    for key in sorted(set(per_task) | set(overall)):
        v = seed_value(per_task, overall, key)
        if v is not None:
            cells.setdefault((key[0], key[1]), []).append(v)

    labels = sorted({k[0] for k in cells})
    suites = sorted({k[1] for k in cells})
    print("| policy | " + " | ".join(suites) + " |")
    print("|---" * (len(suites) + 1) + "|")
    for label in labels:
        row = [label]
        for suite in suites:
            v = cells.get((label, suite))
            if not v:
                row.append("—")
            elif len(v) == 1:
                row.append(f"{v[0]:.1f}")
            else:
                row.append(f"{statistics.mean(v):.1f} ± {statistics.stdev(v):.1f}")
        print("| " + " | ".join(row) + " |")

    if args.per_task:
        suite = args.per_task
        print(f"\nPer-task ({suite}):")
        tids = sorted({t for (l, s, _), ts in per_task.items() if s == suite for t in ts})
        print("| task_id | " + " | ".join(labels) + " |")
        print("|---" * (len(labels) + 1) + "|")
        for tid in tids:
            row = [str(tid)]
            for label in labels:
                vals = [ts[tid] for (l, s, _), ts in per_task.items()
                        if l == label and s == suite and tid in ts]
                row.append(f"{statistics.mean(vals):.0f}" if vals else "—")
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
