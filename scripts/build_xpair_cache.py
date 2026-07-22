"""Offline builder: encode every LIBERO episode (ALL frames; original + pre-rendered
sim-recolor / camera-jitter variants) into the ragged trajectory cache
(tokens.mmap + index.json). build_records/make_chunked are pure and unit-tested.

  python scripts/build_xpair_cache.py --encoder dino   --out outputs/xpair_cache/dino \
      --rendered_dir outputs/rendered_recolor
  python scripts/build_xpair_cache.py --encoder vjepa2 --out outputs/xpair_cache/vjepa2 \
      --rendered_dir outputs/rendered_recolor

Sharded build (one GPU per shard, then a cheap CPU merge; the merged cache is
byte-identical to an unsharded build):
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/build_xpair_cache.py \
      --encoder vjepa2 --out outputs/xpair_cache/vjepa2 --shard $i --num_shards 4 & done; wait
  python scripts/build_xpair_cache.py --encoder vjepa2 --out outputs/xpair_cache/vjepa2 \
      --merge_shards 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.cache_io import CacheHeader, write_cache
from src.traj_data.encoder import build_traj_encoder, encoder_format


def make_chunked(encode, chunk: int, even: bool = False):
    """Encode long clips in fixed-size temporal chunks and concat along tokens.
    even=True trims the clip to an even frame count first (vjepa2 tubelet=2)."""
    def _enc(clip):                                        # (1,T,C,H,W)
        t = clip.shape[1]
        if even and t % 2:
            clip, t = clip[:, : t - 1], t - 1
        outs = [encode(clip[:, i:i + chunk]) for i in range(0, t, chunk)]
        return torch.cat(outs, dim=1)
    return _enc


def build_records(episodes, encode, encoder_id: str, fmt: str, aug_set: str,
                  extra_variants=None):
    """episodes yield {"frames": (T,C,H,W) in [0,1], "instruction": str,
    "episode": int, "task_index": int}. encode maps (1,T,C,H,W) -> (1,L,D).
    extra_variants(episode_id) may supply [(variant_id, frames (T',C,H,W))]."""
    seqs, records, task_texts = [], [], {}

    def _emit(ep, clip, variant):
        toks = encode(clip.unsqueeze(0))[0].cpu().numpy().astype(np.float16)
        seqs.append(toks)
        records.append({"episode": ep["episode"], "variant": variant,
                        "task_index": ep["task_index"]})

    for ep in episodes:
        task_texts.setdefault(int(ep["task_index"]), ep["instruction"])
        _emit(ep, ep["frames"], 0)
        if extra_variants is not None:
            for variant, clip in extra_variants(ep["episode"]):
                _emit(ep, clip, variant)
    header = CacheHeader(encoder_id=encoder_id, format=fmt, d_enc=int(seqs[0].shape[-1]),
                         aug_set=aug_set, num_records=len(records))
    return seqs, records, header, task_texts


def merge_shards(out_dir: str, shard_dirs: list) -> tuple:
    """Concatenate per-shard ragged caches into one at out_dir, re-sorted by
    (episode, variant) so the result equals an unsharded build. Streaming
    mmap-to-mmap copy; headers must agree across shards."""
    import json

    metas = []
    for d in shard_dirs:
        with open(os.path.join(d, "index.json")) as fh:
            metas.append(json.load(fh))
    hdr = dict(metas[0]["header"])
    for m in metas[1:]:
        h = m["header"]
        if (h["encoder_id"], h["format"], h["d_enc"], h["aug_set"]) != \
                (hdr["encoder_id"], hdr["format"], hdr["d_enc"], hdr["aug_set"]):
            raise ValueError(f"shard headers disagree: {h} vs {hdr}")
    entries = []                                           # (ep, var, task, shard, off, len)
    for si, m in enumerate(metas):
        for r in m["records"]:
            entries.append((r["episode"], r["variant"], r["task_index"], si,
                            r["offset"], r["length"]))
    entries.sort(key=lambda e: (e[0], e[1]))
    d_enc = int(hdr["d_enc"])
    total = sum(e[5] for e in entries)
    os.makedirs(out_dir, exist_ok=True)
    mm = np.memmap(os.path.join(out_dir, "tokens.mmap"), dtype=np.float16,
                   mode="w+", shape=(total, d_enc))
    shard_mm = [np.memmap(os.path.join(d, "tokens.mmap"), dtype=np.float16, mode="r",
                          shape=(sum(r["length"] for r in m["records"]), d_enc))
                for d, m in zip(shard_dirs, metas)]
    off, records = 0, []
    for ep, var, ti, si, soff, length in entries:
        mm[off:off + length] = shard_mm[si][soff:soff + length]
        records.append({"episode": ep, "variant": var, "task_index": ti,
                        "offset": off, "length": length})
        off += length
    mm.flush()
    del mm
    task_texts = {}
    for m in metas:
        for k, v in m.get("task_texts", {}).items():
            task_texts.setdefault(k, v)
    hdr["num_records"] = len(records)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"header": hdr, "records": records, "task_texts": task_texts}, fh)
    return len(records), total


def _load_episodes(repo_id, revision, shard=0, num_shards=1):  # pragma: no cover (GPU/dataset)
    """Yield full episodes from lerobot/libero (LeRobotDataset 0.5.1) — ALL frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import prefetch_all_data_parquets

    prefetch_all_data_parquets(repo_id, revision)
    ds = LeRobotDataset(repo_id, revision=revision)
    img_key = ds.meta.camera_keys[0]
    eps = ds.meta.episodes

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    acc = 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        if row.get("dataset_from_index") is not None:
            start = int(row["dataset_from_index"])
            length = int(row["dataset_to_index"]) - start
        else:
            start, length = acc, int(row["length"])
            acc += length
        if ep % num_shards != shard:
            continue
        frames = torch.stack([ds[start + i][img_key] for i in range(length)])
        first = ds[start]
        yield {"frames": frames, "instruction": first.get("task", ""),
               "episode": ep, "task_index": int(first["task_index"])}


def main(argv=None):  # pragma: no cover (GPU)
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--encoder", default="dino")            # dino | vjepa2
    p.add_argument("--encoder_model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--rendered_dir", default=None,
                   help="dir of ep*.npz rendered variants (render_recolor_clips.py)")
    p.add_argument("--chunk", type=int, default=0,
                   help="temporal encode chunk (0 = auto: dino 64, vjepa2 32)")
    p.add_argument("--vjepa_grid", type=int, default=2,
                   help="vjepa2: s×s spatial tokens per tubelet (1 = legacy mean-pool)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1,
                   help=">1: encode episodes ep%%num_shards==shard into <out>.shard<i>")
    p.add_argument("--merge_shards", type=int, default=0,
                   help="merge <out>.shard0..N-1 into <out> (no GPU) and exit")
    args = p.parse_args(argv)

    if args.merge_shards:
        shards = [f"{args.out}.shard{i}" for i in range(args.merge_shards)]
        n, total = merge_shards(args.out, shards)
        print(f"merged {args.merge_shards} shards -> {n} records ({total} tokens) at {args.out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, encode = build_traj_encoder(args.encoder, args.encoder_model, device,
                                   vjepa_grid=args.vjepa_grid)
    chunk = args.chunk or (32 if args.encoder == "vjepa2" else 64)
    encode = make_chunked(encode, chunk, even=(args.encoder == "vjepa2"))

    extra_variants, aug_set = None, "orig"
    if args.rendered_dir:
        rendered: dict[int, list] = {}
        for path in sorted(Path(args.rendered_dir).glob("ep*.npz")):
            with np.load(path) as z:
                rendered.setdefault(int(z["episode"]), []).append((str(z["color"]), path))
        tags = sorted({c for lst in rendered.values() for c, _ in lst})
        aug_set = "orig+" + "+".join(tags)

        def extra_variants(ep_id):
            out = []
            for tag, path in rendered.get(ep_id, []):
                with np.load(path) as z:
                    arr = z["frames"]                      # (T,H,W,C) uint8
                clip = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
                out.append((1 + tags.index(tag), clip))
            return out

        print(f"rendered variants: {sum(len(v) for v in rendered.values())} clips, tags={tags}")

    out_dir = (f"{args.out}.shard{args.shard}" if args.num_shards > 1 else args.out)
    episodes = _load_episodes(args.repo_id, args.revision, args.shard, args.num_shards)
    seqs, records, header, task_texts = build_records(
        episodes, encode, args.encoder,
        encoder_format(args.encoder, args.vjepa_grid), aug_set,
        extra_variants=extra_variants)
    write_cache(out_dir, seqs, records, header, task_texts)
    print(f"wrote {header.num_records} records "
          f"({sum(s.shape[0] for s in seqs)} tokens) to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
