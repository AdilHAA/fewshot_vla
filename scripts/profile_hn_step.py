"""Where does a training step go on the patches arm? Splits the cost into cache
read, hypernetwork attention, and everything else, at the real sequence length.

  python scripts/profile_hn_step.py --cache outputs/xpair_cache/dino_raw
  python scripts/profile_hn_step.py --cache outputs/xpair_cache/dino --batch 32

No lerobot, no SmolVLA: it replays exactly the two things that scale with the clip
length -- the mmap read that `read_row` does in the MAIN process, and the HN's
self-attention over [32 layer-query | 48 text | clip]. Everything else in the step
is independent of the arm, so the difference between arms lives here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, n, dev, warmup=2):
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--layer_queries", type=int, default=32)
    p.add_argument("--text", type=int, default=48)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from src.traj_data.traj_cache import TrajCache

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = TrajCache(args.cache)
    hdr = cache.header
    lens = np.array([r["length"] for r in cache.records])
    rng = np.random.default_rng(args.seed)

    print(f"cache   {args.cache}")
    print(f"  format={hdr['format']} tokens_per_unit={hdr['tokens_per_unit']} "
          f"d_enc={hdr['d_enc']} records={len(lens)}")
    print(f"  tokens per record: median={int(np.median(lens))} mean={lens.mean():.0f} "
          f"max={lens.max()}")

    # A real batch: `batch` random records, padded to the longest one -- exactly what
    # pack_conditioning does.
    rows = rng.choice(len(lens), size=args.batch, replace=False)
    Lmax = int(lens[rows].max())
    L = args.layer_queries + args.text + Lmax
    mb = lens[rows].sum() * int(hdr["d_enc"]) * 2 / 2**20
    print(f"\nbatch of {args.batch}: padded clip len={Lmax}, HN seq len={L}, "
          f"cache read={mb:.0f} MB/step")

    # 1) the synchronous mmap read
    def read():
        for r in rows:
            cache.read_row(int(r))
    t_read = timed(read, args.iters, dev, warmup=1)

    # 2) the HN self-attention at that exact length
    enc = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=args.hidden, nhead=args.heads, dim_feedforward=2 * args.hidden,
            batch_first=True, norm_first=True),
        num_layers=args.blocks).to(dev)
    seq = torch.randn(args.batch, L, args.hidden, device=dev, requires_grad=True)
    pad = torch.zeros(args.batch, L, dtype=torch.bool, device=dev)
    for b, r in enumerate(rows):                       # True == pad, as in the policy
        pad[b, args.layer_queries + args.text + int(lens[r]):] = True

    def hn(mask):
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            out = enc(seq, src_key_padding_mask=mask)
            out[:, : args.layer_queries].square().sum().backward()
        seq.grad = None

    t_hn = timed(lambda: hn(pad), args.iters, dev)
    t_hn_nopad = timed(lambda: hn(None), args.iters, dev)

    flops = 4 * L * L * args.hidden * args.blocks * args.batch * 3   # fwd+bwd, attn only
    print(f"\n  cache read (main process) : {t_read*1000:8.1f} ms")
    print(f"  HN fwd+bwd, with pad mask : {t_hn*1000:8.1f} ms   "
          f"({flops/t_hn/1e12:.1f} TFLOPS on attention alone)")
    print(f"  HN fwd+bwd, mask=None     : {t_hn_nopad*1000:8.1f} ms   "
          f"<- what length-bucketing would unlock")
    print(f"  --> per step, these two   : {(t_read+t_hn)*1000:8.1f} ms "
          f"= {(t_read+t_hn)*100000/3600:.1f} h over 100k steps")
    if t_hn_nopad > 0:
        print(f"      dropping the pad mask alone: x{t_hn/t_hn_nopad:.2f}")


if __name__ == "__main__":
    main()
