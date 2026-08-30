"""Split generator tests (T1): determinism, stratification, subtask policy.
Run: python3 tests/test_libero90_split.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.make_libero90_split import build_split, main

TASKS = json.load(open(os.path.join(ROOT, "configs/libero90_tasks.json")))
SUB_IDS = {i for t in TASKS["libero_10"] for i in t["libero_90_subtask_ids"]}


def _ids(split, part):
    return [t["task_id"] for t in split[part]]


def test_registry_snapshot_shape():
    assert len(TASKS["libero_90"]) == 90 and len(TASKS["libero_10"]) == 10
    texts = {t["instruction"] for t in TASKS["libero_90"]}
    assert len(texts) == 74                       # 12 texts are shared by 28 tasks
    twins = [t for t in TASKS["libero_90"] if t["twin_ids"]]
    assert len(twins) == 28
    for t in twins:                               # twins share text, never scene
        for j in t["twin_ids"]:
            o = TASKS["libero_90"][j]
            assert o["instruction"] == t["instruction"] and o["scene"] != t["scene"]
    assert len({t["scene"] for t in TASKS["libero_90"]}) == 20
    assert len(SUB_IDS) == 14


def test_split_partition_and_determinism():
    a = build_split(TASKS, 42, 40, True, "train")
    b = build_split(TASKS, 42, 40, True, "train")
    assert a == b                                  # deterministic under a seed
    tr, ev = _ids(a, "train"), _ids(a, "eval")
    assert len(tr) == 40 and len(ev) == 50
    assert sorted(tr + ev) == list(range(90))      # exact disjoint partition
    c = build_split(TASKS, 7, 40, True, "train")
    assert _ids(c, "train") != tr                  # the seed matters


def test_stratification_and_subtask_policy():
    a = build_split(TASKS, 42, 40, True, "train")
    assert a["summary"]["n_scenes_covered_by_train"] == 20
    assert SUB_IDS <= set(_ids(a, "train"))        # LIBERO-10 substeps all seen
    e = build_split(TASKS, 42, 40, True, "eval")
    assert SUB_IDS <= set(_ids(e, "eval"))         # ...or all held-out on demand
    assert e["summary"]["n_scenes_covered_by_train"] == 20


def test_main_writes_committed_config():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "split.json")
        split = main(["--out", out])
        assert json.load(open(out)) == split
    committed = os.path.join(ROOT, "configs/libero90_split.json")
    if os.path.exists(committed):                  # the checked-in file IS the default run
        assert json.load(open(committed)) == split


RUN = [test_registry_snapshot_shape, test_split_partition_and_determinism,
       test_stratification_and_subtask_policy, test_main_writes_committed_config]
if __name__ == "__main__":
    for fn in RUN:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL SPLIT TESTS PASS")
