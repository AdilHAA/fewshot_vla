"""Offline builder: demo episodes -> Qwen3.5 VIDEO tokens, one pass, no train-time ViT.

The Qwen3.5 vision tower attends per frame (frames never mix inside the ViT), and its
merger outputs LLM-dim tokens, so the visual token sequence of a demo is a pure
function of its frames. This encodes every episode ONCE into the same ragged cache
format the dino/vjepa arms already use; training then runs only the frozen text stack.

Temporal coverage: every `--every`-th frame, first to last — a FIXED INTERVAL, so
every episode is conditioned at the SAME temporal resolution and the token count
scales with clip length (a fixed frame budget would silently give long episodes a
worse resolution than short ones). The same index function derives the scratch
control's strided dino CLS cache, so the trunk arm and its control see bit-identical
frames.

  python scripts/build_qwen35_video_cache.py --out outputs/xpair_cache/qwen35vl_e4 \
      --every 4                       # shard with --shard/--num_shards as usual
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_xpair_cache import _load_episodes
from src.traj_data.cache_io import CacheHeader, CacheWriter
from src.traj_data.encoder import default_model_id  # noqa: F401 (reuse pattern)
from src.traj_data.stride import stride_indices

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
# Frames are resized to this square before the processor so the token count per frame
# is pinned (14x14 patches -> 7x7 merged = 49 tokens/frame); without a pin the
# processor's smart_resize picks its own resolution and the cache size drifts.
FRAME_SIZE = 224


def encode_frames(model, processor, frames: torch.Tensor, every: int):
    """frames (T,C,H,W) in [0,1] -> (tokens (L, d_llm) fp16, n_frames_used).

    Keeps every `every`-th frame (first to last included) — a fixed INTERVAL, so the
    temporal resolution is identical for every episode; the token count then scales
    with clip length. Resizes to FRAME_SIZE and runs the frozen vision tower + merger
    (no_grad: nothing inside is trained)."""
    idx = stride_indices(frames.shape[0], every)
    sel = frames[list(idx)]                                   # (N,C,H,W)
    sel = torch.nn.functional.interpolate(
        sel, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False)
    vid = (sel.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)   # (N,H,W,3)
    pv = processor.video_processor(videos=[vid], num_frames=len(idx), fps=None,
                                   return_tensors="pt")
    pvv = pv["pixel_values_videos"].to(model.device, next(model.parameters()).dtype)
    grid = pv["video_grid_thw"].to(model.device)
    with torch.no_grad():
        vfeat = model.get_video_features(pixel_values_videos=pvv, video_grid_thw=grid)
    toks = torch.cat(vfeat.pooler_output, dim=0)              # (L, d_llm)
    return toks.to(torch.float16).cpu().numpy(), len(idx)


def main(argv=None):  # pragma: no cover (GPU/weights)
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--every", type=int, default=4,
                   help="keep every k-th frame (fixed interval: same temporal "
                        "resolution for every episode; token count scales with length)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(argv)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model)

    episodes = _load_episodes(args.repo_id, args.revision, args.shard,
                              args.num_shards, workers=args.workers)
    out_dir = (f"{args.out}.shard{args.shard}" if args.num_shards > 1 else args.out)
    d_enc = int(model.config.text_config.hidden_size)
    fmt = f"qwen35vl_every{args.every}"
    writer, task_texts, n = None, {}, 0
    for ep in episodes:
        task_texts.setdefault(int(ep["task_index"]), ep["instruction"])
        toks, used = encode_frames(model, processor, ep["frames"], args.every)
        if writer is None:
            writer = CacheWriter(out_dir, int(toks.shape[-1]))
        writer.add(toks, {"episode": ep["episode"], "variant": 0,
                          "task_index": ep["task_index"], "n_frames": used})
        n += 1
        if n % 100 == 0:
            print(f"[shard {args.shard}] encoded {n} episodes", flush=True)
    if writer is None:
        raise RuntimeError("no episodes produced")
    header = CacheHeader(encoder_id="qwen35vl", format=fmt, d_enc=d_enc, aug_set="orig",
                         num_records=0, chunk=0, encoder_model=args.model,
                         tokens_per_unit=49)         # 7x7 merged tokens per temporal pair @224px
    header.stride = args.every
    total = writer.close(header, task_texts)
    print(f"wrote {len(writer.records)} records ({total} tokens, format={fmt}) "
          f"to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
