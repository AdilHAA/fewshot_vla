"""Bring a LIBERO-90 lerobot v3.0 root to the exact conventions of lerobot/libero, in place,
so the two can be merged (lerobot_edit_dataset merge) and trained/evaled together:

  fps 20 -> 10 (same frames; NVIDIA labels the LIBERO stream 20 Hz, lerobot/libero 10 Hz):
      info.json fps + video.fps, data `timestamp` x2, meta/episodes from/to_timestamp and
      timestamp stats x2, stats.json timestamp x2, every mp4 remuxed with doubled pts
      (stream copy, no re-encode, pixels untouched)
  drop observation.states.{ee_state,joint_state,gripper_state} (absent in lerobot/libero)
  feature `names` and robot_type as in lerobot/libero

  python scripts/harmonize_libero90.py --root outputs/libero90/libero_90_lerobot_v3
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DROP = ("observation.states.ee_state", "observation.states.joint_state", "observation.states.gripper_state")
NAMES = {"observation.state": ["state"], "action": ["actions"],
         "observation.images.image": ["height", "width", "channel"],
         "observation.images.image2": ["height", "width", "channel"]}
SCALE = 2


def _scaled(values):
    return (np.asarray(values, dtype=np.float64) * SCALE).tolist()


def fix_info(root):
    p = root / "meta/info.json"
    info = json.loads(p.read_text())
    if info["fps"] == 10:
        raise SystemExit("fps is already 10 — nothing to do")
    assert info["fps"] == 20, info["fps"]
    info["fps"] = 10.0
    info["robot_type"] = "panda"
    for k in DROP:
        info["features"].pop(k, None)
    for k, names in NAMES.items():
        info["features"][k]["names"] = names
        info["features"][k]["fps"] = 10.0
        if info["features"][k]["dtype"] == "video":
            info["features"][k]["info"]["video.fps"] = 10
    p.write_text(json.dumps(info, indent=4) + "\n")
    return info


def fix_stats(root):
    p = root / "meta/stats.json"
    stats = json.loads(p.read_text())
    for k in DROP:
        stats.pop(k, None)
    ts = stats["timestamp"]
    for k in ("mean", "std", "min", "max"):
        ts[k] = _scaled(ts[k])
    p.write_text(json.dumps(stats, indent=2) + "\n")


def fix_parquet(pf, data):
    t = pq.read_table(pf)
    drop = [c for c in t.schema.names if any(c == k or c.startswith(f"stats/{k}/") for k in DROP)]
    t = t.drop_columns(drop) if drop else t
    cols = ["timestamp"] if data else [c for c in t.schema.names
                                       if c.endswith(("/from_timestamp", "/to_timestamp"))
                                       or c.startswith("stats/timestamp/")]
    for c in cols:
        f = t.schema.field(c)
        vals = t[c].to_pylist()
        if pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type):
            new = [_scaled(v) if v is not None else None for v in vals]
        else:
            new = [v * SCALE if v is not None else None for v in vals]
        t = t.set_column(t.schema.get_field_index(c), f, pa.array(new, type=f.type))
    pq.write_table(t, pf)


def remux(path):
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(path)) as i, av.open(str(tmp), "w") as o:
        vs = i.streams.video[0]
        cc = vs.codec_context
        # any AV1 encoder name only labels the copied stream; packets are muxed as is
        enc = next(n for n in ("libsvtav1", "libaom-av1", "av1") if n in av.codecs_available)
        os_ = o.add_stream(enc, rate=10)
        os_.width, os_.height, os_.pix_fmt = cc.width, cc.height, cc.pix_fmt
        os_.codec_context.extradata = cc.extradata
        os_.time_base = vs.time_base
        for p in i.demux(vs):
            if p.dts is None:
                continue
            p.pts *= SCALE
            p.dts *= SCALE
            if p.duration:
                p.duration *= SCALE
            p.stream = os_
            o.mux(p)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    av.logging.set_level(av.logging.ERROR)

    fix_info(root)
    fix_stats(root)
    for pf in sorted(root.glob("data/**/*.parquet")):
        fix_parquet(pf, data=True)
    for pf in sorted(root.glob("meta/episodes/**/*.parquet")):
        fix_parquet(pf, data=False)
    videos = sorted(root.glob("videos/**/*.mp4"))
    for n, v in enumerate(videos, 1):
        remux(v)
        print(f"remuxed {n}/{len(videos)} {v.relative_to(root)}", flush=True)

    info = json.loads((root / "meta/info.json").read_text())
    t = pq.read_table(sorted(root.glob("data/**/*.parquet"))[0], columns=["timestamp", "frame_index"]).slice(0, 3).to_pylist()
    with av.open(str(videos[0])) as c:
        rate = c.streams.video[0].average_rate
    print(f"fps={info['fps']} robot_type={info['robot_type']} features={sorted(info['features'])}")
    print(f"first timestamps={[r['timestamp'] for r in t]} video avg_rate={rate}")


if __name__ == "__main__":
    main()
