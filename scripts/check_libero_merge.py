"""Prove Kesvill/libero_90_lerobot_v3 is mergeable with lerobot/libero: run lerobot's own
metadata check (the one `merge` uses), then optionally merge for real and read a frame
from each half of the result.

  python scripts/check_libero_merge.py                 # metadata check only
  python scripts/check_libero_merge.py --merge         # + build outputs/libero90/libero_all and open it
"""
import argparse
from pathlib import Path

from lerobot.datasets.aggregate import validate_all_metadata
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPOS = ["lerobot/libero", "Kesvill/libero_90_lerobot_v3"]


def describe(ds, i):
    x = ds[i]
    return (f"episode {int(x['episode_index'])} frame {int(x['frame_index'])} t={x['timestamp'].item():.2f} "
            f"image {tuple(x['observation.images.image'].shape)} gripper {x['action'][-1].item():+.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--out", default="outputs/libero90/libero_all")
    args = p.parse_args()

    datasets = [LeRobotDataset(r, video_backend="pyav") for r in REPOS]
    for d in datasets:
        print(f"{d.repo_id}: fps={d.meta.fps} robot={d.meta.robot_type} episodes={d.meta.total_episodes} "
              f"frames={d.meta.total_frames} features={len(d.meta.features)}")
    fps, robot, features = validate_all_metadata([d.meta for d in datasets])
    print(f"validate_all_metadata OK: fps={fps} robot_type={robot} features={sorted(features)}")
    if not args.merge:
        return

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists — remove it first")
    merged = merge_datasets(datasets, "local/libero_all", out)
    n = sum(d.meta.total_episodes for d in datasets)
    assert merged.meta.total_episodes == n, (merged.meta.total_episodes, n)
    print(f"merged: episodes={merged.meta.total_episodes} frames={merged.meta.total_frames} "
          f"tasks={len(merged.meta.tasks)} -> {out}")
    print("first frame (lerobot/libero side):", describe(merged, 0))
    print("last frame  (libero_90 side):     ", describe(merged, len(merged) - 1))


if __name__ == "__main__":
    main()
