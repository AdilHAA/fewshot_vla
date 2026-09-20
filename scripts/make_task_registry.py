"""configs/task_registry.json: LIBERO bddl stem (task_key) -> suite, task order, text.

WHY a registry instead of the dataset's own task ids: the merged HN dataset rebuilds
`task_index` from instruction TEXT, so it collapses the 28 LIBERO-90 tasks that share
a sentence across scenes (and 2 texts across the two source datasets) — a task_index
is NOT a task identity. The bddl stem is: it equals `LiberoEnv.task` (`task.name`),
all 130 stems of the 5 suites are unique, and LIBERO-Pro's libero_10_{lan,object,
swap,task} reuse the base libero_10 stems verbatim, so an exact stem match resolves a
perturbed rollout to the task whose demos condition it.

  python scripts/make_task_registry.py                 # needs libero installed
  python scripts/make_task_registry.py --from_configs  # no libero: checked-in snapshot
  python scripts/make_task_registry.py --check         # exit 1 if the file drifted

task_id is the task's order WITHIN its suite, i.e. the index lerobot-eval's
--env.task_ids addresses. libero_90/libero_10 ids come from configs/libero90_tasks.json
(a checked-in snapshot of libero_task_map); goal/object/spatial from BASE_STEMS below,
a snapshot of libero_suite_task_map.py in LIBERO's benchmark order. `--check` on a
machine with libero installed proves both snapshots against the installed package.
Nothing but eval task selection depends on task_id — task_key is the identity.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "configs/task_registry.json")
TASKS_JSON = os.path.join(ROOT, "configs/libero90_tasks.json")

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_90")
EXPECTED_COUNTS = {"libero_10": 10, "libero_goal": 10, "libero_object": 10,
                   "libero_spatial": 10, "libero_90": 90}
SCENE_RE = re.compile(r"^(.*_SCENE\d+)_")

SOURCE_LIBERO = ("libero.libero.benchmark (get_benchmark_dict, 5 suites, 130 bddl "
                 "stems); key = task.name, instruction = task.language, task_id = "
                 "benchmark order = lerobot-eval --env.task_ids order")
SOURCE_CONFIGS = ("configs/libero90_tasks.json (libero_90/libero_10) + libero_suite_task_map "
                  "snapshot (goal/object/spatial), task_id = LIBERO benchmark order = "
                  "lerobot-eval --env.task_ids order; instruction = "
                  "grab_language_from_filename(stem)")

# The three suites configs/libero90_tasks.json does not cover (it snapshots
# libero_90 + libero_10 only), in libero_suite_task_map order == task_id order.
BASE_STEMS = {
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
}


def scene_of(key: str) -> str:
    m = SCENE_RE.match(key)
    return m.group(1) if m else ""


def language_of(stem: str) -> str:
    """LIBERO's grab_language_from_filename: '_' -> ' ', minus the SCENE prefix the
    LIBERO-100 suites carry (verified to reproduce all 100 snapshot instructions)."""
    scene = scene_of(stem)
    return (stem[len(scene) + 1:] if scene else stem).replace("_", " ")


def _finish(rows: list, source: str) -> dict:
    counts: dict = {}
    for r in rows:
        counts[r["suite"]] = counts.get(r["suite"], 0) + 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"suite task counts {counts} != {EXPECTED_COUNTS}")
    keys = [r["key"] for r in rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"task keys must be unique across suites; duplicated: {dupes}")
    rows.sort(key=lambda r: (SUITES.index(r["suite"]), r["task_id"]))
    return {"source": source, "tasks": rows}


def build_from_libero() -> dict:
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    rows = []
    for suite in SUITES:
        b = bench[suite]()
        for task_id in range(b.n_tasks):
            t = b.get_task(task_id)
            rows.append({"key": t.name, "suite": suite, "task_id": task_id,
                         "instruction": t.language, "scene": scene_of(t.name)})
    return _finish(rows, SOURCE_LIBERO)


def build_from_configs(tasks_path: str = TASKS_JSON) -> dict:
    with open(tasks_path) as fh:
        tasks = json.load(fh)
    rows = []
    for suite in ("libero_10", "libero_90"):
        for t in tasks[suite]:
            rows.append({"key": t["bddl_name"], "suite": suite,
                         "task_id": int(t["task_id"]), "instruction": t["instruction"],
                         "scene": t["scene"]})
    for suite, stems in BASE_STEMS.items():
        for task_id, stem in enumerate(stems):
            rows.append({"key": stem, "suite": suite, "task_id": task_id,
                         "instruction": language_of(stem), "scene": scene_of(stem)})
    return _finish(rows, SOURCE_CONFIGS)


def check_pro_stems(rows: list) -> list:
    """LIBERO-Pro's libero_10_* suites reuse the base libero_10 bddl stems verbatim —
    that is what lets an exact stem match resolve a perturbed rollout to its base task.
    Checked against the installed bddl files; silently skipped if Pro is not installed."""
    from libero.libero import get_libero_path

    base = {r["key"] for r in rows if r["suite"] == "libero_10"}
    root = Path(get_libero_path("bddl_files"))
    checked = []
    for suite in (f"libero_10_{axis}" for axis in ("lan", "object", "swap", "task")):
        stems = {p.stem for p in (root / suite).glob("*.bddl")}
        if not stems:
            continue
        if stems != base:
            raise ValueError(f"{suite} stems differ from libero_10: "
                             f"+{sorted(stems - base)} -{sorted(base - stems)}")
        checked.append(suite)
    return checked


def diff_tasks(have: list, want: list) -> list:
    a = {r["key"]: r for r in have}
    b = {r["key"]: r for r in want}
    out = [f"only in the committed file: {k}" for k in sorted(set(a) - set(b))]
    out += [f"missing from the committed file: {k}" for k in sorted(set(b) - set(a))]
    for k in sorted(set(a) & set(b)):
        for f in ("suite", "task_id", "instruction", "scene"):
            if a[k].get(f) != b[k].get(f):
                out.append(f"{k}: {f} {a[k].get(f)!r} != rebuilt {b[k].get(f)!r}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--from_configs", action="store_true",
                   help="build from the checked-in snapshots (no libero needed)")
    p.add_argument("--check", action="store_true",
                   help="compare a fresh build with --out; exit 1 on any difference")
    args = p.parse_args(argv)

    reg = build_from_configs() if args.from_configs else build_from_libero()
    if not args.from_configs:
        pro = check_pro_stems(reg["tasks"])
        print(f"LIBERO-Pro stem check: {', '.join(pro) if pro else 'skipped (not installed)'}")

    if args.check:
        with open(args.out) as fh:
            have = json.load(fh)
        diff = diff_tasks(have["tasks"], reg["tasks"])
        if diff:
            print(f"{args.out} differs from a fresh build:")
            for line in diff:
                print("  " + line)
            sys.exit(1)
        print(f"{args.out} matches a fresh build ({len(reg['tasks'])} tasks)")
        return

    with open(args.out, "w") as fh:
        json.dump(reg, fh, indent=1)
        fh.write("\n")
    per_suite = {s: sum(1 for r in reg["tasks"] if r["suite"] == s) for s in SUITES}
    print(f"wrote {len(reg['tasks'])} tasks to {args.out}: {per_suite}")


if __name__ == "__main__":
    main()
