"""nvidia/LIBERO_LeRobot_v3 (libero_90, OpenVLA no-op filtered) -> ready for finetune_ours.sh.

download -> wrist_image->image2 -> gripper 0/1 -> LIBERO -1/+1 (data + stats) ->
episode->task_id by matching action sequences against yzembodied ->
train/eval episode lists for configs/libero90_split.json -> orientation check
of the first frames against the flipped yzembodied dataset.

  python scripts/prepare_nvidia_libero90.py --out outputs/libero90/nvidia \\
      --reference outputs/libero90/libero_90_image_flipped
"""
import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.rename_libero90_features import rename_json, rename_parquet  # noqa: E402

REPO = "nvidia/LIBERO_LeRobot_v3"
REV = "e5907374380b8f96511957e6ba5582be52a1e179"
RENAME = {"observation.images.wrist_image": "observation.images.image2"}
GRIP = 6


def download(out):
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", revision=REV,
                      allow_patterns=["libero_90/**"], local_dir=str(out))
    return out / "libero_90"


def rename_features(root):
    for f in ("info.json", "stats.json"):
        rename_json(root / "meta" / f, RENAME)
    for pf in list(root.glob("meta/episodes/**/*.parquet")) + list(root.glob("data/**/*.parquet")):
        rename_parquet(pf, RENAME)
    for old, new in RENAME.items():
        if (root / "videos" / old).is_dir():
            (root / "videos" / old).rename(root / "videos" / new)


def _flip_stats(s):
    # g' = 1 - 2g: mean/std scale, min/max swap
    mean, std, lo, hi = (np.array(s[k], dtype=np.float64) for k in ("mean", "std", "min", "max"))
    mean[GRIP], std[GRIP] = 1 - 2 * mean[GRIP], 2 * std[GRIP]
    lo[GRIP], hi[GRIP] = 1 - 2 * hi[GRIP], 1 - 2 * lo[GRIP]
    return {**s, "mean": mean.tolist(), "std": std.tolist(), "min": lo.tolist(), "max": hi.tolist()}


def convert_gripper(root):
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    if stats["action"]["min"][GRIP] < 0:
        print("gripper already in LIBERO convention, skipping")
        return
    for pf in sorted(root.glob("data/**/*.parquet")):
        t = pq.read_table(pf)
        field = t.schema.field("action")
        a = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        assert np.isin(a[:, GRIP], (0.0, 1.0)).all(), f"{pf}: gripper not in {{0,1}}"
        a[:, GRIP] = 1 - 2 * a[:, GRIP]
        t = t.set_column(t.schema.get_field_index("action"), field, pa.array(a.tolist(), type=field.type))
        pq.write_table(t, pf)
    stats["action"] = _flip_stats(stats["action"])
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    for pf in root.glob("meta/episodes/**/*.parquet"):
        t = pq.read_table(pf)
        cols = {k: f"stats/action/{k}" for k in ("mean", "std", "min", "max")}
        if not all(c in t.schema.names for c in cols.values()):
            continue
        rows = [_flip_stats({k: t[c][i].as_py() for k, c in cols.items()}) for i in range(t.num_rows)]
        for k, c in cols.items():
            idx = t.schema.get_field_index(c)
            t = t.set_column(idx, t.schema.field(c), pa.array([r[k] for r in rows], type=t.schema.field(c).type))
        pq.write_table(t, pf)
    print("gripper converted: 0/1 -> +1 (close) / -1 (open)")


def read_actions(root):
    grouped = defaultdict(list)
    for pf in sorted(Path(root).glob("data/**/*.parquet")):
        t = pq.read_table(pf, columns=["action", "episode_index", "frame_index"])
        a = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        ep, fr = t["episode_index"].to_numpy(), t["frame_index"].to_numpy()
        for e in np.unique(ep):
            m = ep == e
            grouped[int(e)].append((fr[m], a[m]))
    out = {}
    for e, parts in grouped.items():
        fr = np.concatenate([p[0] for p in parts])
        a = np.concatenate([p[1] for p in parts])
        out[e] = a[np.argsort(fr)]
    return out


def drop_noops(a):
    # OpenVLA regenerate_libero_dataset.py: |motion| < 1e-4 and gripper unchanged vs last kept action
    keep, prev = [], None
    for i, x in enumerate(a):
        if np.linalg.norm(x[:GRIP]) < 1e-4 and (prev is None or x[GRIP] == prev):
            continue
        keep.append(i)
        prev = x[GRIP]
    return a[keep]


def key(a):
    a = np.array(a, dtype="<f4")
    a[a == 0] = 0
    return hashlib.sha256(a.tobytes()).hexdigest()


def map_tasks(root, reference, split_path):
    ref = {key(drop_noops(a)): e for e, a in read_actions(reference).items()}
    mapping, ref_ep, unmatched = {}, {}, []
    for e, a in read_actions(root).items():
        r = ref.get(key(a))
        if r is None:
            unmatched.append(e)
            continue
        mapping[e], ref_ep[e] = r // 50, r
    if unmatched:
        raise SystemExit(f"{len(unmatched)} NVIDIA episodes matched no yzembodied episode: {unmatched[:10]}")
    split = json.loads(Path(split_path).read_text())
    present = set(mapping.values())
    lists = {}
    for part in ("train", "eval"):
        ids = {t["task_id"] for t in split[part]}
        missing = sorted(ids - present)
        lists[part] = sorted(e for e, t in mapping.items() if t in ids)
        (root / f"{part}_episodes.json").write_text(json.dumps(lists[part]) + "\n")
        print(f"{part}: {len(ids) - len(missing)}/{len(ids)} tasks, {len(lists[part])} episodes"
              f"{', MISSING tasks ' + str(missing) if missing else ''}")
        if part == "eval" and missing:
            raise SystemExit("eval tasks missing from the NVIDIA data — split would change")
    (root / "episode_task_map.json").write_text(json.dumps(
        {"revision": REV, "reference": str(Path(reference).resolve()),
         "episode_to_task": mapping, "episode_to_reference_episode": ref_ep}, indent=1) + "\n")
    return ref_ep


def _nvidia_frame0(root, ep, cam="observation.images.image"):
    import av

    for pf in root.glob("meta/episodes/**/*.parquet"):
        t = pq.read_table(pf).filter(pq.read_table(pf)["episode_index"] == ep)
        if t.num_rows:
            row = t.slice(0, 1).to_pylist()[0]
            break
    else:
        raise KeyError(ep)
    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["video_path"].format(video_key=cam, chunk_index=row[f"videos/{cam}/chunk_index"],
                                            file_index=row[f"videos/{cam}/file_index"])
    ts = row[f"videos/{cam}/from_timestamp"]
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(ts / float(s.time_base)), stream=s)
        for fr in c.decode(s):
            if fr.time is None or fr.time >= ts - 1e-3:
                return fr.to_ndarray(format="rgb24")
    raise RuntimeError(f"no frame at {ts}s in {path}")


def _reference_frame0(reference, ep, cam="observation.images.image"):
    from PIL import Image

    for pf in sorted(Path(reference).glob("data/**/*.parquet")):
        t = pq.read_table(pf, columns=[cam, "episode_index", "frame_index"],
                          filters=[("episode_index", "==", ep), ("frame_index", "==", 0)])
        if t.num_rows:
            return np.asarray(Image.open(io.BytesIO(t[cam][0].as_py()["bytes"])).convert("RGB"))
    raise KeyError(ep)


def check_orientation(root, reference, ref_ep, n=3):
    tf = {"identity": lambda x: x, "rot180": lambda x: x[::-1, ::-1], "fliplr": lambda x: x[:, ::-1],
          "flipud": lambda x: x[::-1]}
    bad = 0
    for e in sorted(ref_ep)[:n]:
        nv = _nvidia_frame0(root, e).astype(np.float32)
        rf = _reference_frame0(reference, ref_ep[e]).astype(np.float32)
        mae = {k: float(np.abs(f(nv) - rf).mean()) for k, f in tf.items()}
        best = min(mae, key=mae.get)
        bad += best != "identity"
        print(f"orientation nvidia ep {e} vs reference ep {ref_ep[e]}: "
              + ", ".join(f"{k}={v:.1f}" for k, v in mae.items()) + f" -> {best}")
    if bad:
        print("WARNING: NVIDIA frames are NOT in the reference (policy-input) orientation — do not train as is")
    else:
        print("orientation OK: NVIDIA frames match the flipped yzembodied / policy-input convention")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/libero90/nvidia")
    p.add_argument("--reference", default="outputs/libero90/libero_90_image_flipped",
                   help="yzembodied root (flipped one: its frames are the policy-input reference)")
    p.add_argument("--split", default="configs/libero90_split.json")
    p.add_argument("--skip_download", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    root = out / "libero_90" if args.skip_download else download(out)
    rename_features(root)
    convert_gripper(root)
    ref_ep = map_tasks(root, args.reference, args.split)
    check_orientation(root, args.reference, ref_ep)
    print(f"dataset root: {root}\ntrain list: {root / 'train_episodes.json'}")


if __name__ == "__main__":
    main()
