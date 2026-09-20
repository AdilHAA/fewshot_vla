"""<dataset root>/episode_keys.json: dataset episode_index -> LIBERO bddl stem.

WHY: the merged dataset (lerobot/libero + Kesvill/libero_90_lerobot_v3) rebuilds
`task_index` from instruction TEXT, so its task_index merges tasks that share a
sentence — 28 LIBERO-90 tasks across scenes, plus 2 texts that exist in both halves.
Conditioning pools must be keyed by the bddl stem (configs/task_registry.json)
instead. The cache builders read this sidecar to stamp every record with its
task_key; eval matches the same stem against `LiberoEnv.task`.

Resolution is per half:
  * LIBERO-90 (episode_index >= --libero90_offset): the Kesvill episode->task_id map
    is authoritative — it is the only thing that survives the twins — and the
    episode's own text is asserted against the registry instruction.
  * lerobot/libero: instruction text, matched against the FOUR base suites only.
    'turn on the stove' and 'pick up the book and place it in the back compartment of
    the caddy' are ALSO LIBERO-90 task texts, so an unrestricted text match would be
    ambiguous exactly where it matters.

  python scripts/make_episode_keys.py --root outputs/libero90/libero_all \
      --libero90_map <kesvill root>/episode_task_map.json --libero90_offset 1693
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.traj_cache import norm_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(ROOT, "configs/task_registry.json")
BASE_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def index_registry(registry: dict) -> tuple:
    """(base-suite text -> [keys], libero_90 task_id -> row). Texts are normalized."""
    by_text: dict = {}
    by_task_id: dict = {}
    for r in registry["tasks"]:
        if r["suite"] in BASE_SUITES:
            by_text.setdefault(norm_text(r["instruction"]), []).append(r["key"])
        elif r["suite"] == "libero_90":
            by_task_id[int(r["task_id"])] = r
    return by_text, by_task_id


def episode_rows(table) -> list:
    """(episode_index, instruction) rows of one meta/episodes table.

    lerobot v3 stores `tasks` as a one-element list of strings per episode; older
    dumps use a scalar `task`. Accepts a pyarrow Table or a list of row dicts."""
    rows = table.to_pylist() if hasattr(table, "to_pylist") else list(table)
    out = []
    for r in rows:
        text = r["tasks"] if "tasks" in r else r["task"]
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"episode {r['episode_index']}: {len(text)} task "
                                 f"strings {text}; one episode is one task here")
            text = text[0]
        out.append((int(r["episode_index"]), str(text)))
    return out


def build_episode_keys(rows, registry: dict, libero90_map: dict | None = None,
                       libero90_offset: int = 0) -> dict:
    """rows: (episode_index, instruction) pairs -> {episode_index: task_key}."""
    by_text, by_task_id = index_registry(registry)
    ep_to_task = (libero90_map or {}).get("episode_to_task", {})
    out: dict = {}
    for ep, text in rows:
        ep = int(ep)
        if libero90_map is not None and ep >= libero90_offset:
            src = str(ep - libero90_offset)
            if src not in ep_to_task:
                raise KeyError(f"episode {ep} (source episode {src}) is not in the "
                               f"LIBERO-90 episode_to_task map")
            row = by_task_id[int(ep_to_task[src])]
            if norm_text(text) != norm_text(row["instruction"]):
                raise ValueError(f"episode {ep}: dataset text {text!r} != registry "
                                 f"{row['instruction']!r} for {row['key']} — the map "
                                 f"and the dataset disagree on this episode")
            out[ep] = row["key"]
        else:
            hits = by_text.get(norm_text(text), [])
            if len(hits) != 1:
                raise ValueError(f"episode {ep}: text {text!r} matches {len(hits)} "
                                 f"base-suite tasks {hits}")
            out[ep] = hits[0]
    return out


def read_episode_rows(root: str) -> list:  # pragma: no cover (needs a dataset)
    """All (episode_index, instruction) rows of a lerobot v3 dataset root."""
    import pyarrow.parquet as pq

    paths = sorted(Path(root, "meta", "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no meta/episodes/**/*.parquet under {root}")
    rows = []
    for path in paths:
        # Only these columns: the episode tables also carry per-feature image stats,
        # which are orders of magnitude larger than everything else.
        names = [c for c in ("episode_index", "tasks", "task")
                 if c in pq.read_schema(path).names]
        rows += episode_rows(pq.read_table(path, columns=names))
    rows.sort()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="lerobot dataset root")
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--libero90_map", default=None,
                   help="Kesvill episode_task_map.json (episode_to_task); required "
                        "for the LIBERO-90 half of a merged dataset")
    p.add_argument("--libero90_offset", type=int, default=1693,
                   help="merged episode_index of Kesvill episode 0")
    p.add_argument("--out", default=None, help="default: <root>/episode_keys.json")
    args = p.parse_args(argv)

    with open(args.registry) as fh:
        registry = json.load(fh)
    l90 = None
    if args.libero90_map:
        with open(args.libero90_map) as fh:
            l90 = json.load(fh)

    rows = read_episode_rows(args.root)
    keys = build_episode_keys(rows, registry, l90, args.libero90_offset)

    suite_of = {r["key"]: r["suite"] for r in registry["tasks"]}
    per_suite: dict = {}
    tasks_per_suite: dict = {}
    for key in keys.values():
        s = suite_of[key]
        per_suite[s] = per_suite.get(s, 0) + 1
        tasks_per_suite.setdefault(s, set()).add(key)

    sources = [f"root={args.root}", f"registry={args.registry}"]
    if l90 is not None:
        sources.append(f"libero90_map={args.libero90_map} "
                       f"(offset={args.libero90_offset}, "
                       f"reference={l90.get('reference', '?')}, "
                       f"revision={str(l90.get('revision', '?'))[:8]})")
    out = args.out or os.path.join(args.root, "episode_keys.json")
    with open(out, "w") as fh:
        json.dump({"episode_to_key": {str(k): v for k, v in sorted(keys.items())},
                   "n_episodes": len(keys), "sources": sources}, fh)
    print(f"wrote {len(keys)} episodes / {len(set(keys.values()))} tasks to {out}")
    for s in sorted(per_suite):
        print(f"  {s:<14} {per_suite[s]:>5} episodes  {len(tasks_per_suite[s]):>3} tasks")


if __name__ == "__main__":
    main()
