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

from src.traj_data.cache_io import (HEADER_DEFAULTS, SHARD_KEYS, CacheHeader,
                                    CacheWriter, write_cache)
from src.traj_data.encoder import (DINO_SIDE, build_traj_encoder, default_model_id,
                                   derive_tokens, encoder_format, parse_format)


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


def iter_records(episodes, encode, extra_variants=None, task_texts=None):
    """Yield (tokens (L,d) fp16, record) per episode/variant, one at a time.

    A generator, not a list: the raw-patch build is 108 GB of tokens and must never
    be held in memory. `task_texts`, if given, is filled in as a side effect."""
    def _emit(ep, clip, variant):
        toks = encode(clip.unsqueeze(0))[0].cpu().numpy().astype(np.float16)
        return toks, {"episode": ep["episode"], "variant": variant,
                      "task_index": ep["task_index"]}

    for ep in episodes:
        if task_texts is not None:
            task_texts.setdefault(int(ep["task_index"]), ep["instruction"])
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
    import json

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


def _load_episodes(repo_id, revision, shard=0, num_shards=1,
                   workers=8):  # pragma: no cover (GPU/dataset)
    """Yield full episodes from lerobot/libero (LeRobotDataset 0.5.1) — ALL frames.

    Per-frame `ds[i]` access random-seeks into the episode videos and decodes one
    frame at a time — the actual bottleneck of a full-frame build. Frames are
    therefore streamed through a DataLoader with `workers` parallel decoders
    (order-preserving), and sliced back into episodes here."""
    from collections import deque

    from torch.utils.data import DataLoader, Subset

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import prefetch_all_data_parquets

    prefetch_all_data_parquets(repo_id, revision)
    ds = LeRobotDataset(repo_id, revision=revision)
    img_key = ds.meta.camera_keys[0]
    eps = ds.meta.episodes

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    all_eps, plan, acc = [], [], 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        if row.get("dataset_from_index") is not None:
            start = int(row["dataset_from_index"])
            length = int(row["dataset_to_index"]) - start
        else:
            start, length = acc, int(row["length"])
            acc += length
        all_eps.append((ep, start, length))

    # Balance shards by FRAME COUNT, not by episode count: LIBERO episodes run
    # 75..505 frames, so round-robin `ep % num_shards` leaves one H100 grinding
    # while the others idle. Longest-processing-time-first greedy assignment gets
    # every shard within ~one episode of the mean.
    if num_shards > 1:
        load = [0] * num_shards
        owner = {}
        for ep, _, length in sorted(all_eps, key=lambda e: -e[2]):
            s = min(range(num_shards), key=lambda i: load[i])
            owner[ep], load[s] = s, load[s] + length
        plan = [e for e in all_eps if owner[e[0]] == shard]
        if not plan:
            # An empty shard means the id is out of range (e.g. --shard 2 --num_shards
            # 2: valid ids are 0..num_shards-1) — say so instead of dying later with
            # a bare IndexError from the episode deque.
            raise ValueError(
                f"shard {shard} owns 0 episodes: num_shards={num_shards} assigns to "
                f"shards 0..{num_shards - 1} (loads {load}). If you meant one shard "
                f"per GPU, pass BOTH the gpu index in CUDA_VISIBLE_DEVICES and the "
                f"shard index in 0..{num_shards - 1}.")
        print(f"[shard {shard}] {len(plan)} episodes, {sum(e[2] for e in plan)} frames "
              f"(shard loads: {load})", flush=True)
    else:
        plan = all_eps

    idxs = [i for _, start, length in plan for i in range(start, start + length)]
    # Decode is the build's bottleneck (per-frame random access into the episode
    # videos), so keep many workers busy and a deep prefetch queue: the encoder must
    # never wait on a decoder. persistent_workers avoids re-forking per epoch.
    loader = DataLoader(Subset(ds, idxs), batch_size=256, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        prefetch_factor=(4 if workers else None),
                        persistent_workers=bool(workers))

    pending = deque(plan)
    cur_ep, _, cur_need = pending.popleft()
    cur_frames, cur_meta, done = [], None, 0
    for batch in loader:
        imgs = batch[img_key]
        tasks = batch.get("task")
        tis = batch["task_index"]
        for j in range(imgs.shape[0]):
            if cur_meta is None:
                cur_meta = (tasks[j] if tasks is not None else "", int(tis[j]))
            cur_frames.append(imgs[j])
            if len(cur_frames) == cur_need:
                done += 1
                if done % 20 == 0 or done == len(plan):
                    print(f"[shard {shard}] decoded {done}/{len(plan)} episodes",
                          flush=True)
                yield {"frames": torch.stack(cur_frames), "instruction": cur_meta[0],
                       "episode": cur_ep, "task_index": cur_meta[1]}
                cur_frames, cur_meta = [], None
                if pending:
                    cur_ep, _, cur_need = pending.popleft()
    if cur_frames or pending:
        raise RuntimeError(f"decode stream ended early: {len(pending)} episodes left")


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
                   help=">1: encode episodes ep%%num_shards==shard into <out>.shard<i>")
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
                              args.num_shards, workers=args.workers)
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
