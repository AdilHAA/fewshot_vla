"""Prove Kesvill/libero_90_lerobot_v3 is mergeable with lerobot/libero: run lerobot's own
metadata check (the one `merge` uses), then optionally merge for real and read a frame
from each half of the result.

  python scripts/check_libero_merge.py                 # metadata check only
  python scripts/check_libero_merge.py --merge         # + build outputs/libero90/libero_all and open it

lerobot/libero was converted by an early v3.0 writer: its meta/episodes table has no
`tasks` and no `meta/episodes/{chunk,file}_index` columns (merge reads both), and its
`data/file_index` does not match the files' contents. The merge therefore reads it
through a local shadow root (data/ and videos/ symlinked, episodes table completed
from the data parquets) — the Hub copy is never modified.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets import aggregate
from lerobot.datasets.aggregate import validate_all_metadata
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPOS = ["lerobot/libero", "Kesvill/libero_90_lerobot_v3"]


def shadow_root(src: Path, dst: Path) -> Path:
    """A root that lerobot's merge can read: src's data/videos, a completed episodes table."""
    eps = sorted(src.glob("meta/episodes/*/*.parquet"))
    assert len(eps) == 1, f"expected one episodes file in {src}, got {len(eps)}"
    df = pd.read_parquet(eps[0])
    if "tasks" in df and "meta/episodes/chunk_index" in df:
        return src
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "videos"):
        if not (dst / sub).exists():
            os.symlink((src / sub).resolve(), dst / sub)
    (dst / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    for f in ("info.json", "stats.json", "tasks.parquet"):
        shutil.copyfile(src / "meta" / f, dst / "meta" / f)

    files = sorted(src.glob("data/*/*.parquet"))
    loc, ep_task = {}, {}
    for f in files:
        chunk, file = int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1])
        tb = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for e, ti in tb.drop_duplicates().itertuples(index=False):
            assert loc.setdefault(int(e), (chunk, file)) == (chunk, file), f"episode {e} in two files"
            ep_task.setdefault(int(e), set()).add(int(ti))
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    text_of = {int(ti): name for name, ti in zip(tasks.index, tasks["task_index"])}
    e = df["episode_index"].astype(int)
    if "tasks" not in df:
        df["tasks"] = [[text_of[ti] for ti in sorted(ep_task[i])] for i in e]
    df["data/chunk_index"] = [loc[i][0] for i in e]
    df["data/file_index"] = [loc[i][1] for i in e]
    df["meta/episodes/chunk_index"] = 0
    df["meta/episodes/file_index"] = 0
    df.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    print(f"shadow root {dst}: episodes table completed ({len(df)} rows, {len(files)} data files)")
    return dst


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
    if args.merge:
        fixed = shadow_root(Path(datasets[0].meta.root), Path(args.out).parent / "lerobot_libero_shadow")
        if fixed != Path(datasets[0].meta.root):
            datasets[0] = LeRobotDataset(REPOS[0], root=fixed, video_backend="pyav")
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
    # lerobot seeds every merged video file with shutil.copy, which carries the
    # source's mode bits over: hub-cache files are read-only, so appending the next
    # source file (write into that copy) fails. Copy without the mode (process-wide
    # patch; this CLI does nothing else). The temp mp4 of each append is then renamed
    # into place, so keep it on the output filesystem, not /tmp.
    aggregate.shutil.copy = lambda src, dst: (shutil.copyfile(src, dst), os.chmod(dst, 0o644))[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = tempfile.mkdtemp(prefix="merge_tmp_", dir=out.parent)
    merged = merge_datasets(datasets, "local/libero_all", out)
    shutil.rmtree(tempfile.tempdir, ignore_errors=True)
    n = sum(d.meta.total_episodes for d in datasets)
    assert merged.meta.total_episodes == n, (merged.meta.total_episodes, n)
    print(f"merged: episodes={merged.meta.total_episodes} frames={merged.meta.total_frames} "
          f"tasks={len(merged.meta.tasks)} -> {out}")
    print("first frame (lerobot/libero side):", describe(merged, 0))
    print("last frame  (libero_90 side):     ", describe(merged, len(merged) - 1))


if __name__ == "__main__":
    main()
