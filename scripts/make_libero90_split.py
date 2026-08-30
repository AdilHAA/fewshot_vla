"""LIBERO-90 train/eval split: 40 train / 50 held-out, fixed in git (T1 of the
held-out stage, docs/experiments/2026-08-30-plan-libero90-heldout.md).

Reads configs/libero90_tasks.json — a checked-in snapshot of LIBERO's
libero_task_map (task_id = list order = the order lerobot-eval's --env.task_ids
addresses) — so the split is reproducible WITHOUT libero installed. On a machine
that has libero, --verify_against_libero re-derives the registry and asserts the
snapshot matches it byte-for-byte.

Split policy (all deterministic under --seed):
  * scene-stratified (default): each of the 20 scenes contributes >=1 train task,
    so every held-out task is "unseen task in a SEEN scene", never "unseen scene"
    — otherwise the 90-eval column would silently mix two transfer distances.
  * --libero10_subtasks train (default): the 14 LIBERO-90 tasks that are literal
    sub-steps of LIBERO-10 tasks (see libero_90_subtask_ids in the tasks file)
    are forced into train, so the LIBERO-10 column reads "long-horizon
    composition of SEEN subtasks" instead of an unlabeled train/eval mixture.
    `eval` forces them held-out instead; `free` applies no constraint.

Twins (12 instruction texts shared by 28 tasks across scenes) are NOT
constrained — but twin_ids ride along in the output so 90-eval can later be
reported stratified by instruction-seen/unseen.

  python scripts/make_libero90_split.py                  # writes configs/libero90_split.json
  python scripts/make_libero90_split.py --seed 7 --out /tmp/alt.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_split(tasks: dict, seed: int, n_train: int, stratify: bool,
                libero10_subtasks: str) -> dict:
    t90 = tasks["libero_90"]
    assert [t["task_id"] for t in t90] == list(range(len(t90)))
    sub_ids = sorted({i for t in tasks["libero_10"] for i in t["libero_90_subtask_ids"]})
    rng = random.Random(seed)

    train: set[int] = set()
    banned: set[int] = set()
    if libero10_subtasks == "train":
        train |= set(sub_ids)
    elif libero10_subtasks == "eval":
        banned |= set(sub_ids)

    if stratify:
        by_scene: dict[str, list[int]] = {}
        for t in t90:
            by_scene.setdefault(t["scene"], []).append(t["task_id"])
        for scene in sorted(by_scene):
            if train & set(by_scene[scene]):
                continue
            pool = sorted(set(by_scene[scene]) - banned)
            if not pool:
                raise ValueError(f"scene {scene}: every task is banned from train")
            train.add(rng.choice(pool))

    pool = sorted(set(range(len(t90))) - train - banned)
    if len(train) > n_train:
        raise ValueError(f"constraints already force {len(train)} > n_train={n_train}")
    train |= set(rng.sample(pool, n_train - len(train)))
    eval_ids = sorted(set(range(len(t90))) - train)

    def rows(ids):
        return [t90[i] for i in sorted(ids)]

    scenes_train = {t["scene"] for i in train for t in [t90[i]]}
    split_twins = sorted({tuple(sorted([t["task_id"]] + t["twin_ids"]))
                          for t in t90 if t["twin_ids"]
                          and any(i in train for i in [t["task_id"]] + t["twin_ids"])
                          and any(i not in train for i in [t["task_id"]] + t["twin_ids"])})
    return {
        "seed": seed,
        "policy": {"n_train": n_train, "stratify_by_scene": stratify,
                   "libero10_subtasks": libero10_subtasks,
                   "forced_subtask_ids": sub_ids},
        "train": rows(train),
        "eval": rows(eval_ids),
        "summary": {
            "n_train": len(train), "n_eval": len(eval_ids),
            "scenes_covered_by_train": sorted(scenes_train),
            "n_scenes_covered_by_train": len(scenes_train),
            "twin_groups_split_across_train_eval": [list(g) for g in split_twins],
        },
    }


def verify_against_libero(tasks: dict) -> None:  # pragma: no cover (needs libero)
    import re

    from libero.libero.benchmark import libero_suite_task_map as m
    for suite, key in [("libero_90", "libero_90"), ("libero_10", "libero_10")]:
        names = m.libero_task_map[key]
        ours = [t["bddl_name"] for t in tasks[suite]]
        assert names == ours, f"{suite}: snapshot diverges from installed libero"
        for t, name in zip(tasks[suite], names):
            sc = re.match(r"^(.*_SCENE\d+)_", name).group(1)
            assert t["scene"] == sc and t["instruction"] == name[len(sc) + 1:].replace("_", " ")
    print("snapshot matches installed libero registry")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=os.path.join(ROOT, "configs/libero90_tasks.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "configs/libero90_split.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--no-stratify", dest="stratify", action="store_false",
                   help="pure random split (default: every scene is seen in train)")
    p.add_argument("--libero10_subtasks", choices=["train", "eval", "free"],
                   default="train",
                   help="where the 14 LIBERO-10 sub-step tasks must land (default train)")
    p.add_argument("--verify_against_libero", action="store_true",
                   help="assert the checked-in registry snapshot matches installed libero")
    args = p.parse_args(argv)

    with open(args.tasks) as fh:
        tasks = json.load(fh)
    if args.verify_against_libero:
        verify_against_libero(tasks)
    split = build_split(tasks, args.seed, args.n_train, args.stratify,
                        args.libero10_subtasks)
    with open(args.out, "w") as fh:
        json.dump(split, fh, indent=1, ensure_ascii=False)
    s = split["summary"]
    print(f"wrote {args.out}: {s['n_train']} train / {s['n_eval']} eval, "
          f"{s['n_scenes_covered_by_train']}/20 scenes in train, "
          f"{len(s['twin_groups_split_across_train_eval'])} twin groups split")
    return split


if __name__ == "__main__":
    main()
