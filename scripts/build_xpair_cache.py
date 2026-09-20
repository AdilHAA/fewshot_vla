"""Offline builder: encode every LIBERO episode (ALL frames; original + pre-rendered
sim-recolor / camera-jitter variants) into the ragged trajectory cache
(tokens.mmap + index.json). build_records/make_chunked are pure and unit-tested.

  python scripts/build_xpair_cache.py --encoder dino   --out outputs/xpair_cache/dino \
      --rendered_dir outputs/rendered_recolor --legacy_keys
  python scripts/build_xpair_cache.py --encoder vjepa2 --out outputs/xpair_cache/vjepa2 \
      --rendered_dir outputs/rendered_recolor --legacy_keys

Merged LIBERO-90 + lerobot/libero dataset (local root, task identity from the
episode_keys sidecar rather than the dataset's text-derived task_index):
  python scripts/build_xpair_cache.py --encoder dino --out outputs/xpair_cache/dino90 \
      --repo_id local/libero_all --root outputs/libero90/libero_all \
      --video_backend pyav --keys outputs/libero90/libero_all/episode_keys.json \
      --episodes outputs/libero90/hn_train_episodes.json

Sharded build (one GPU per shard, then a cheap CPU merge; the merged cache is
byte-identical to an unsharded build):
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/build_xpair_cache.py \
      --encoder vjepa2 --out outputs/xpair_cache/vjepa2 --shard $i --num_shards 4 \
      --legacy_keys & done; wait
  python scripts/build_xpair_cache.py --encoder vjepa2 --out outputs/xpair_cache/vjepa2 \
      --merge_shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.cache_io import (HEADER_DEFAULTS, SHARD_KEYS, CacheHeader,
                                    CacheWriter, write_cache)
from src.traj_data.encoder import (DINO_SIDE, build_traj_encoder, default_model_id,
                                   derive_tokens, encoder_format, parse_format)


def add_dataset_args(p):
    """Dataset-source flags shared by every cache builder (build_qwen35_video_cache
    imports these too, so the two can never drift on which episodes they encode)."""
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--root", default=None,
                   help="local LeRobotDataset root (e.g. the merged libero_all "
                        "build); the hub revision is then not used")
    p.add_argument("--video_backend", default=None,
                   help="lerobot decoder; 'pyav' for the AV1 LIBERO videos")
    p.add_argument("--episodes", default=None,
                   help="json file: episode_index list the build is restricted to")
    p.add_argument("--keys", default=None,
                   help="episode_keys.json (scripts/make_episode_keys.py): "
                        "episode_index -> LIBERO bddl stem, stamped on every record")
    p.add_argument("--legacy_keys", action="store_true",
                   help="write pre-registry task_index records instead of task_key "
                        "ones (only valid where task_index IS the task, i.e. the "
                        "unmerged lerobot/libero dataset)")


def dataset_kwargs(args) -> dict:
    """The shared source flags as `_load_episodes` kwargs."""
    if not args.legacy_keys and not args.keys:
        raise SystemExit(
            "--keys <episode_keys.json> is required: the merged dataset's task_index "
            "is rebuilt from instruction TEXT and merges same-sentence tasks. Build "
            "one with scripts/make_episode_keys.py, or pass --legacy_keys.")
    if args.legacy_keys and (args.keys or args.root):
        raise SystemExit(
            "--legacy_keys is only valid for the unmerged Hub lerobot/libero dataset "
            "(where task_index IS the task); a local --root is a merged build — pass "
            "--keys instead, not both.")
    keys = None
    if not args.legacy_keys:
        with open(args.keys) as fh:
            keys = {int(k): v for k, v in json.load(fh)["episode_to_key"].items()}
    eps = None
    if args.episodes:
        with open(args.episodes) as fh:
            eps = [int(e) for e in json.load(fh)]
    return {"root": args.root, "video_backend": args.video_backend,
            "episodes": eps, "keys": keys}


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


def _task_of(ep):
    """(record task fields, task_texts key) of a source episode.

    A registry build stamps the bddl stem (`task_key`); a --legacy_keys build keeps
    the dataset's `task_index`, which is only a task identity on a dataset that was
    not merged (the merged one rebuilds task_index from instruction text)."""
    k = ep.get("task_key")
    if k is None:
        return {"task_index": int(ep["task_index"])}, int(ep["task_index"])
    return {"task_key": str(k)}, str(k)


def iter_records(episodes, encode, extra_variants=None, task_texts=None):
    """Yield (tokens (L,d) fp16, record) per episode/variant, one at a time.

    A generator, not a list: the raw-patch build is 108 GB of tokens and must never
    be held in memory. `task_texts`, if given, is filled in as a side effect.
    `src_len` is the frame count of the clip behind the record — the trunk arm needs
    it to recompute the frame indices (and hence the prompt timestamps) of a strided
    encode without touching the dataset again."""
    def _emit(ep, clip, variant):
        toks = encode(clip.unsqueeze(0))[0].cpu().numpy().astype(np.float16)
        fields, _ = _task_of(ep)
        return toks, {"episode": ep["episode"], "variant": variant,
                      "src_len": int(clip.shape[0]), **fields}

    for ep in episodes:
        if task_texts is not None:
            task_texts.setdefault(_task_of(ep)[1], ep["instruction"])
        yield _emit(ep, ep["frames"], 0)
        if extra_variants is not None:
            for variant, clip in extra_variants(ep["episode"]):
                yield _emit(ep, clip, variant)


def build_records(episodes, encode, encoder_id: str, fmt: str, aug_set: str,
                  extra_variants=None, chunk: int = 0, encoder_model: str = "",
                  tokens_per_unit: int = 0):
    """Collecting wrapper around `iter_records` (small caches, tests). The GPU build
    path streams through CacheWriter instead — same generator, no RAM ceiling."""
    seqs, records, task_texts = [], [], {}
    for toks, rec in iter_records(episodes, encode, extra_variants, task_texts):
        seqs.append(toks)
        records.append(rec)
    header = CacheHeader(encoder_id=encoder_id, format=fmt, d_enc=int(seqs[0].shape[-1]),
                         aug_set=aug_set, num_records=len(records),
                         chunk=int(chunk), encoder_model=encoder_model,
                         tokens_per_unit=int(tokens_per_unit))
    return seqs, records, header, task_texts


def merge_shards(out_dir: str, shard_dirs: list) -> tuple:
    """Concatenate per-shard ragged caches into one at out_dir, re-sorted by
    (episode, variant) so the result equals an unsharded build. Streaming
    mmap-to-mmap copy; headers must agree across shards."""
    metas = []
    for d in shard_dirs:
        with open(os.path.join(d, "index.json")) as fh:
            metas.append(json.load(fh))
    hdr = dict(metas[0]["header"])
    # Every key that changes the MEANING of a token must agree, not just the ones
    # that change its shape: shards built with a different encoder checkpoint or a
    # different encode window would otherwise merge silently into one mmap.
    def _cmp(h, k):
        """Shard-comparable value of one header key. Keys whose pre-2026-08-02
        value is knowable are back-filled; `chunk`/`encoder_model` genuinely are
        not, so a shard that predates them compares as a wildcard (None) instead
        of being given an invented value — this is what lets a single re-run shard
        merge back into an otherwise older set."""
        return h.get(k, HEADER_DEFAULTS.get(k))

    for m in metas[1:]:
        h = m["header"]
        for k in SHARD_KEYS:
            a, b = _cmp(h, k), _cmp(hdr, k)
            if k in ("chunk", "encoder_model") and (a is None or b is None):
                print(f"WARNING: shard provenance key '{k}' unrecorded in at least "
                      f"one shard ({a!r} vs {b!r}) — cannot cross-check", flush=True)
                continue
            if a != b:
                raise ValueError(f"shard headers disagree on {k}: {a!r} != {b!r}")
        for k in ("chunk", "encoder_model"):              # keep a recorded value
            if hdr.get(k) is None and h.get(k) is not None:
                hdr[k] = h[k]
    # Records are copied WHOLE (only `offset` is rewritten): rebuilding them from a
    # fixed key set silently dropped task_key/src_len/n_frames on every merge.
    entries = []                                           # (ep, var, shard, record)
    for si, m in enumerate(metas):
        for r in m["records"]:
            entries.append((int(r["episode"]), int(r.get("variant", 0)), si, r))
    entries.sort(key=lambda e: (e[0], e[1]))
    d_enc = int(hdr["d_enc"])
    total = sum(int(e[3]["length"]) for e in entries)
    os.makedirs(out_dir, exist_ok=True)
    mm = np.memmap(os.path.join(out_dir, "tokens.mmap"), dtype=np.float16,
                   mode="w+", shape=(total, d_enc))
    shard_mm = [np.memmap(os.path.join(d, "tokens.mmap"), dtype=np.float16, mode="r",
                          shape=(sum(r["length"] for r in m["records"]), d_enc))
                for d, m in zip(shard_dirs, metas)]
    off, records = 0, []
    for _, _, si, r in entries:
        soff, length = int(r["offset"]), int(r["length"])
        mm[off:off + length] = shard_mm[si][soff:soff + length]
        records.append({**r, "offset": off})
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


class _FileStream(torch.utils.data.IterableDataset):
    """Decode whole video FILES sequentially, one worker per file, and yield the
    planned episodes of each as (episode, frames uint8 (T,H,W,3)).

    lerobot's per-frame reader opens the file and decodes from the previous keyframe
    for every frame; on a merged dataset whose mp4s hold hundreds of episodes that is
    a few frames per second. A file decoded once, front to back, is thousands."""

    def __init__(self, files, fps):
        self.files, self.fps = files, float(fps)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for i, (path, spans) in enumerate(self.files):
            if i % nw == wid:
                yield from _decode_spans(path, spans, self.fps)


def _decode_spans(path, spans, fps):
    """spans: [(episode, from_ts, to_ts, length)] of one file, any order. Runs of
    adjacent episodes are decoded in one pass; a gap (episodes of another shard, or
    not in --episodes) is skipped with a keyframe seek."""
    import av

    tol = 0.5 / fps
    spans = sorted(spans, key=lambda s: s[1])
    runs = []
    for sp in spans:
        if runs and sp[1] - runs[-1][-1][2] <= 2.0:
            runs[-1].append(sp)
        else:
            runs.append([sp])

    def emit(span, frames):
        ep, from_ts, to_ts, length = span
        if len(frames) != length:
            raise RuntimeError(f"episode {ep}: decoded {len(frames)} frames for "
                               f"[{from_ts:.2f}, {to_ts:.2f}) in {path}, meta says {length}")
        return ep, np.stack(frames)

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        for run in runs:
            c.seek(int(max(run[0][1] - tol, 0.0) / tb), stream=s, backward=True, any_frame=False)
            todo, i, frames = run, 0, []
            for fr in c.decode(s):
                if fr.pts is None:
                    continue
                t = fr.pts * tb
                while t >= todo[i][2] - tol:          # past the end of episode i
                    yield emit(todo[i], frames)
                    frames, i = [], i + 1
                    if i == len(todo):
                        break
                if i == len(todo):
                    break
                if t >= todo[i][1] - tol:
                    frames.append(fr.to_ndarray(format="rgb24"))
            else:                                     # file ended inside episode i
                yield emit(todo[i], frames)
                i += 1
            if i != len(todo):
                raise RuntimeError(f"{path} ended with {len(todo) - i} planned episodes undecoded")


def _episode_task_index(root):
    """episode_index -> task_index straight from the data parquets (a meta table's
    data/file_index can be wrong — lerobot/libero's is — so read every file)."""
    import pyarrow.parquet as pq

    out = {}
    for f in sorted(Path(root).glob("data/*/*.parquet")):
        tb = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for e, ti in tb.drop_duplicates("episode_index").itertuples(index=False):
            out.setdefault(int(e), int(ti))
    return out


def _load_episodes(repo_id, revision, shard=0, num_shards=1, workers=8,
                   root=None, video_backend=None, episodes=None,
                   keys=None):  # pragma: no cover (GPU/dataset)
    """Yield full episodes from a lerobot dataset (LeRobotDataset 0.5.1) — ALL frames
    of the first camera, decoded file by file (see _FileStream).

    `root` reads a LOCAL dataset (the merged libero_all build) instead of the hub,
    `episodes` restricts the build to those episode_index values, and `keys` stamps
    each episode with the task_key its records carry. Shards own whole video files,
    balanced by planned frame count, so no file is decoded twice."""
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import DEFAULT_LIBERO_REPO, prefetch_all_data_parquets

    if root is None and repo_id == DEFAULT_LIBERO_REPO:
        # The broken meta file_index this works around is lerobot/libero's alone.
        prefetch_all_data_parquets(repo_id, revision)
    ds = LeRobotDataset(repo_id, root=root, revision=(None if root else revision),
                        video_backend=video_backend)
    img_key = ds.meta.camera_keys[0]
    eps = ds.meta.episodes
    fps = float(ds.meta.fps)
    text_of = {int(ti): str(name) for name, ti in
               zip(ds.meta.tasks.index, ds.meta.tasks["task_index"])}
    task_of = _episode_task_index(ds.meta.root)

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    want = None if episodes is None else {int(e) for e in episodes}
    by_file, n_planned = {}, 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        # `--episodes` and `--keys` address rows by episode_index; a meta table whose
        # row order is not that index would silently encode the wrong episodes.
        if row.get("episode_index") is not None and int(row["episode_index"]) != ep:
            raise ValueError(f"meta/episodes row {ep} carries episode_index "
                             f"{int(row['episode_index'])}: rows are not addressed "
                             f"by episode_index in this dataset")
        if want is not None and ep not in want:
            continue
        if keys is not None and ep not in keys:
            raise KeyError(f"episode {ep} is missing from the episode_keys file; "
                           f"rebuild it for this dataset (scripts/make_episode_keys.py)")
        path = Path(ds.meta.root) / ds.meta.video_path.format(
            video_key=img_key, chunk_index=int(row[f"videos/{img_key}/chunk_index"]),
            file_index=int(row[f"videos/{img_key}/file_index"]))
        span = (ep, float(row[f"videos/{img_key}/from_timestamp"]),
                float(row[f"videos/{img_key}/to_timestamp"]), int(row["length"]))
        by_file.setdefault(path, []).append(span)
        n_planned += 1
    if want is not None and n_planned != len(want):
        have = {s[0] for spans in by_file.values() for s in spans}
        missing = sorted(want - have)
        raise ValueError(f"{len(missing)} requested episodes are not in the dataset "
                         f"(first: {missing[:5]})")
    if not by_file:
        raise ValueError(f"nothing to encode: {ds.meta.total_episodes} episodes in "
                         f"{repo_id}, 0 selected by --episodes")

    # Shards own whole FILES (a file is decoded once, by one process); files are
    # dealt longest-first to the least loaded shard, so shard wall times stay close.
    files = sorted(by_file.items(), key=lambda kv: -sum(s[3] for s in kv[1]))
    load, mine = [0] * num_shards, []
    for path, spans in files:
        s_ = min(range(num_shards), key=lambda i: load[i])
        load[s_] += sum(sp[3] for sp in spans)
        if s_ == shard:
            mine.append((path, spans))
    if not mine:
        raise ValueError(f"shard {shard} owns 0 files: {len(files)} video files over "
                         f"num_shards={num_shards} (loads {load}); use fewer shards")
    plan_n = sum(len(sp) for _, sp in mine)
    print(f"[shard {shard}] {len(mine)} files, {plan_n} episodes, {load[shard]} frames "
          f"(shard loads: {load})", flush=True)

    loader = DataLoader(_FileStream(mine, fps), batch_size=None,
                        num_workers=min(workers, len(mine)),
                        prefetch_factor=(2 if workers else None))
    done = 0
    for ep, frames in loader:
        done += 1
        if done % 20 == 0 or done == plan_n:
            print(f"[shard {shard}] decoded {done}/{plan_n} episodes", flush=True)
        ti = task_of[int(ep)]
        yield {"frames": torch.as_tensor(frames).permute(0, 3, 1, 2).float().div_(255.0),
               "instruction": text_of.get(ti, ""), "episode": int(ep), "task_index": ti,
               "task_key": None if keys is None else keys[int(ep)]}
    if done != plan_n:
        raise RuntimeError(f"decode stream ended early: {done}/{plan_n} episodes")


def main(argv=None):  # pragma: no cover (GPU)
    p = argparse.ArgumentParser()
    add_dataset_args(p)
    p.add_argument("--encoder", default="dino")            # dino | vjepa2
    p.add_argument("--encoder_model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--rendered_dir", default=None,
                   help="dir of ep*.npz rendered variants (render_recolor_clips.py)")
    p.add_argument("--chunk", type=int, default=0,
                   help="vjepa2: temporal encode window in frames (0 = auto 32). This "
                        "is SEMANTIC — it bounds the attention horizon of every token. "
                        "For dino it is only a batch size; use --encode_batch.")
    p.add_argument("--encode_batch", type=int, default=0,
                   help="dino: frames per forward (0 = auto 256 on cuda, 64 on cpu). "
                        "Purely a throughput knob — frames are embedded independently, "
                        "so it cannot change a single token.")
    p.add_argument("--vjepa_grid", type=int, default=2,
                   help="vjepa2: s×s spatial tokens per tubelet (1 = legacy mean-pool)")
    p.add_argument("--dino_patches", action="store_true",
                   help="dino: keep ALL 256 patch tokens of each frame, unpooled")
    p.add_argument("--dino_grid", type=int, default=0,
                   help="dino: block-average the 16x16 patch map to s×s tokens "
                        "(0 = no patch tokens). --dino_patches is the s=16 case, "
                        "where the block is 1x1 and the average is the identity.")
    p.add_argument("--dino_n_reg", type=int, default=0,
                   help="dino: register tokens per frame (dinov2-with-registers only)")
    p.add_argument("--no_cls", action="store_true", help="dino: drop the CLS token")
    p.add_argument("--also_emit", action="append", default=[], metavar="FMT:DIR",
                   help="write a SECOND cache derived from the same forward, e.g. "
                        "--also_emit cls:outputs/xpair_cache/dino. `cls` is token 0 of "
                        "every unit, so the extra cache costs no GPU and no extra "
                        "decode — the decode pass is what a build actually spends.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1,
                   help=">1: split the episodes over <out>.shard<i> (frame-balanced)")
    p.add_argument("--merge_shards", type=int, default=0,
                   help="merge <out>.shard0..N-1 into <out> (no GPU) and exit")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel frame-decode workers (decode is the bottleneck)")
    args = p.parse_args(argv)

    if args.merge_shards:
        shards = [f"{args.out}.shard{i}" for i in range(args.merge_shards)]
        n, total = merge_shards(args.out, shards)
        print(f"merged {args.merge_shards} shards -> {n} records ({total} tokens) at {args.out}")
        return

    source = dataset_kwargs(args)          # before the encoder: fail fast on a typo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_grid = DINO_SIDE if args.dino_patches else args.dino_grid
    fmt = encoder_format(args.encoder, args.vjepa_grid, dino_grid=dino_grid,
                         include_cls=not args.no_cls, n_reg=args.dino_n_reg)
    _, encode = build_traj_encoder(args.encoder, args.encoder_model, device,
                                   vjepa_grid=args.vjepa_grid, dino_grid=dino_grid,
                                   include_cls=not args.no_cls, n_reg=args.dino_n_reg)
    if args.encoder == "vjepa2":
        chunk = args.chunk or 32
    else:
        chunk = args.encode_batch or (256 if device == "cuda" else 64)
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
    episodes = _load_episodes(args.repo_id, args.revision, args.shard,
                              args.num_shards, workers=args.workers, **source)
    header = CacheHeader(
        encoder_id=args.encoder, format=fmt, d_enc=0, aug_set=aug_set, num_records=0,
        # For dino the encode window is a pure batch size — frames are embedded
        # independently — so recording it would pin a value that cannot affect a
        # single token. Only vjepa2's window is semantic (it bounds the temporal
        # attention horizon of every token in it).
        chunk=int(chunk) if args.encoder == "vjepa2" else 0,
        encoder_model=args.encoder_model or default_model_id(args.encoder),
        tokens_per_unit=parse_format(fmt)["tokens_per_unit"])
    extra = []                                   # [(fmt, dir, CacheWriter|None)]
    for spec in args.also_emit:
        efmt, _, edir = spec.partition(":")
        if not edir:
            raise ValueError(f"--also_emit expects FMT:DIR, got {spec!r}")
        derive_tokens(np.zeros((parse_format(fmt)["tokens_per_unit"], 1), np.float16),
                      fmt, efmt)                 # fail fast, before the whole dataset
        extra.append([efmt, (f"{edir}.shard{args.shard}" if args.num_shards > 1 else edir),
                      None])

    task_texts, writer, n = {}, None, 0
    for toks, rec in iter_records(episodes, encode, extra_variants, task_texts):
        if writer is None:
            writer = CacheWriter(out_dir, int(toks.shape[-1]))
        writer.add(toks, rec)
        for e in extra:
            dtoks = np.ascontiguousarray(derive_tokens(toks, fmt, e[0]))
            if e[2] is None:
                e[2] = CacheWriter(e[1], int(dtoks.shape[-1]))
            e[2].add(dtoks, rec)
        n += 1
        if n % 100 == 0:
            print(f"[shard {args.shard}] wrote {n} records", flush=True)
    if writer is None:
        raise RuntimeError("no episodes produced — empty shard?")
    total = writer.close(header, task_texts)
    print(f"wrote {len(writer.records)} records ({total} tokens, format={fmt}) to {out_dir}")
    import dataclasses
    for efmt, edir, ew in extra:
        eh = dataclasses.replace(header, format=efmt,
                                 tokens_per_unit=parse_format(efmt)["tokens_per_unit"])
        et = ew.close(eh, task_texts)
        print(f"  + derived {len(ew.records)} records ({et} tokens, format={efmt}) to {edir}")


if __name__ == "__main__":  # pragma: no cover
    main()
