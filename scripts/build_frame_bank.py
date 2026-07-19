"""Build the first-frame bank: the t=0 frame of every episode (original from
lerobot/libero + rendered sim-recolor / camera-jitter variants).

  python scripts/build_frame_bank.py --out outputs/frame_bank.npz \
      --rendered_dir outputs/rendered_recolor
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):  # pragma: no cover (dataset)
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--out", default="outputs/frame_bank.npz")
    p.add_argument("--rendered_dir", default=None)
    args = p.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import prefetch_all_data_parquets

    prefetch_all_data_parquets(args.repo_id, args.revision)
    ds = LeRobotDataset(args.repo_id, revision=args.revision)
    img_key = ds.meta.camera_keys[0]
    eps = ds.meta.episodes

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    rendered: dict[int, list] = {}
    tags: list = []
    if args.rendered_dir:
        for path in sorted(Path(args.rendered_dir).glob("ep*.npz")):
            with np.load(path) as z:
                rendered.setdefault(int(z["episode"]), []).append((str(z["color"]), path))
        tags = sorted({c for lst in rendered.values() for c, _ in lst})

    images, episode, task_index, variant = [], [], [], []
    task_texts = {}
    acc = 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        start = (int(row["dataset_from_index"])
                 if row.get("dataset_from_index") is not None else acc)
        if row.get("dataset_from_index") is None:
            acc += int(row["length"])
        item = ds[start]
        img = (item[img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ti = int(item["task_index"])
        task_texts.setdefault(ti, item.get("task", ""))
        images.append(img); episode.append(ep); task_index.append(ti); variant.append(0)
        for tag, path in rendered.get(ep, []):
            with np.load(path) as z:
                images.append(z["frames"][0])
            episode.append(ep); task_index.append(ti); variant.append(1 + tags.index(tag))

    import json

    np.savez_compressed(args.out, images=np.stack(images),
                        episode=np.array(episode), task_index=np.array(task_index),
                        variant=np.array(variant),
                        task_texts_json=json.dumps(task_texts))
    print(f"wrote {len(images)} first frames to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
