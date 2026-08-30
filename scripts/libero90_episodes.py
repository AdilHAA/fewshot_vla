"""Episode lists / eval chunks for the LIBERO-90 split (T3/T4 helper).

Assumes the yzembodied-ordered conversion where episode_index // 50 == benchmark
task_id (verified on yzembodied/libero_90_image: 4500 episodes = 90 tasks x 50,
in task order) and NO episode filtering. If a filtered dataset is ever used,
regenerate lists from its episode->task sidecar instead.

  --emit episodes  ->  one python list of dataset episode indices for --dataset.episodes
  --emit chunks    ->  one python list of task_ids PER LINE, for chunked
                       lerobot-eval --env.task_ids (memory: every selected task's
                       MuJoCo env is created eagerly and lives for the whole run)

  python scripts/libero90_episodes.py --part train
  python scripts/libero90_episodes.py --part eval --emit chunks --chunk 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=os.path.join(ROOT, "configs/libero90_split.json"))
    p.add_argument("--part", choices=["train", "eval"], required=True)
    p.add_argument("--emit", choices=["episodes", "chunks"], default="episodes")
    p.add_argument("--per_task", type=int, default=50,
                   help="episodes per task in the dataset (yzembodied: 50)")
    p.add_argument("--chunk", type=int, default=10, help="tasks per eval chunk")
    args = p.parse_args(argv)

    with open(args.split) as fh:
        split = json.load(fh)
    task_ids = [t["task_id"] for t in split[args.part]]

    if args.emit == "episodes":
        eps = [t * args.per_task + k for t in task_ids for k in range(args.per_task)]
        print(json.dumps(eps, separators=(",", ":")))
        print(f"[libero90_episodes] {args.part}: {len(task_ids)} tasks x "
              f"{args.per_task} = {len(eps)} episodes", file=sys.stderr)
    else:
        for i in range(0, len(task_ids), args.chunk):
            print(json.dumps(task_ids[i:i + args.chunk], separators=(",", ":")))


if __name__ == "__main__":
    main()
