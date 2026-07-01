"""Offline builder: encode every LIBERO episode (orig + color variants) into
the trajectory cache (clips.mmap + keys.pt + index.json).

  python scripts/build_xpair_cache.py --out outputs/xpair_cache/dino
  python scripts/build_xpair_cache.py --encoder vjepa2 --num_frames 16 --out outputs/xpair_cache/vjepa2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.augment import color_jitter, opening_clip_indices
from src.traj_data.cache_io import CacheHeader, write_cache
from src.traj_data.encoder import build_traj_encoder
from src.traj_data.retrieval import build_key


def build_records(episodes, encode, sent_encode, num_frames, stride,
                  n_color, beta_t, beta_f, encoder_id, sent_id):
    """Pure orchestration over an iterable of episode dicts. `encode` maps a clip
    (1,T,C,H,W) in [0,1] -> (1, N, D). Returns (clips fp16, keys, records, header)."""
    clips_list, keys_list, records = [], [], []
    d_enc = d_text = n_tokens = None
    row = 0
    for ep in episodes:
        frames = ep["frames"]                                  # (n_avail,C,H,W)
        idx = opening_clip_indices(frames.shape[0], num_frames, stride)
        base_clip = frames[idx]                                # (T,C,H,W)
        text_emb = sent_encode(ep["instruction"])              # (Dtext,)
        for variant in range(1 + n_color):
            clip = base_clip if variant == 0 else color_jitter(
                base_clip, seed=hash((ep["episode"], variant)) & 0x7FFFFFFF)
            toks = encode(clip.unsqueeze(0))[0].cpu()          # (N, D)
            if d_enc is None:
                n_tokens, d_enc = toks.shape[0], toks.shape[-1]
                d_text = text_emb.shape[-1]
            frame_emb = toks.float().mean(dim=0)               # (D,)
            key = build_key(text_emb.float(), frame_emb, beta_t, beta_f)
            clips_list.append(toks.numpy())
            keys_list.append(key)
            records.append({"episode": ep["episode"], "variant": variant,
                            "task_index": ep["task_index"],
                            "object_set": ep["object_set"], "row": row})
            row += 1
    clips = np.stack(clips_list).astype(np.float16)
    keys = torch.stack(keys_list)
    header = CacheHeader(
        encoder_id=encoder_id, num_frames=num_frames, stride=stride, window="opening",
        jitter="chan-gain-bias", d_enc=d_enc, ntok=n_tokens // num_frames,
        n_tokens=n_tokens, sentence_encoder_id=sent_id, beta_t=beta_t, beta_f=beta_f,
        d_text=d_text, d_frame=d_enc, n_clips=row)
    return clips, keys, records, header


def _load_episodes(repo_id, revision, num_frames, stride):  # pragma: no cover (GPU/dataset)
    """Yield opening-clip dicts from lerobot/libero (LeRobotDataset 0.5.1). Only the
    first T frames of each episode are decoded (cheap; leak-free)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import prefetch_all_data_parquets

    prefetch_all_data_parquets(repo_id, revision)          # work around wrong file_index
    ds = LeRobotDataset(repo_id, revision=revision)
    img_key = ds.meta.camera_keys[0]                       # main camera
    eps = ds.meta.episodes                                 # v3.0: dataset_from/to_index cols

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    acc = 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        if row.get("dataset_from_index") is not None:
            start = int(row["dataset_from_index"])
            length = int(row["dataset_to_index"]) - start
        else:                                              # fall back to cumulative length
            start, length = acc, int(row["length"])
            acc += length
        local = opening_clip_indices(length, num_frames, stride)
        frames = torch.stack([ds[start + i][img_key] for i in local])  # (T,C,H,W) in [0,1]
        first = ds[start]
        yield {
            "frames": frames,
            "instruction": first.get("task", ""),
            "episode": ep,
            "task_index": int(first["task_index"]),
            "object_set": str(int(first["task_index"])),
        }


def main(argv=None):  # pragma: no cover (GPU)
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--encoder", default="dino")            # dino | vjepa2
    p.add_argument("--encoder_model", default=None)        # override the HF model id
    p.add_argument("--out", required=True)
    p.add_argument("--num_frames", type=int, default=4)    # vjepa2: use 16 (even)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n_color", type=int, default=2)
    p.add_argument("--sentence_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--beta_t", type=float, default=1.0)
    p.add_argument("--beta_f", type=float, default=1.0)
    args = p.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, encode = build_traj_encoder(args.encoder, args.encoder_model, device)
    st = SentenceTransformer(args.sentence_encoder, device=device)
    sent_encode = lambda s: torch.tensor(st.encode(s), dtype=torch.float32)

    episodes = _load_episodes(args.repo_id, args.revision, args.num_frames, args.stride)
    clips, keys, records, header = build_records(
        episodes, encode, sent_encode, args.num_frames, args.stride,
        args.n_color, args.beta_t, args.beta_f, encoder_id=args.encoder,
        sent_id=args.sentence_encoder)
    write_cache(args.out, clips, keys, records, header)
    print(f"wrote {header.n_clips} clips to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
