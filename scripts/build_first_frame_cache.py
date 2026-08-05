"""Offline builder for the MINIMAL-CLIP trajectory cache: one t=0 frame per
episode -> one record of `grid²` tokens (default 1).

Reads `outputs/frame_bank.npz` (scripts/build_frame_bank.py), which already holds
exactly the frames this arm needs — so there is no lerobot, no LeRobotDataset and
no video decode here at all. The full-clip builder pays ~273k frame decodes; this
one pays zero.

V-JEPA2's patch embed is a Conv3d of depth `tubelet_size=2`, so a single frame is
expanded to the encoder minimum before encoding. `--fill dup` repeats the frame:
[f0, f0] makes the tubelet exactly (W0+W1)·f0, i.e. motion is removed by
construction while the frame content survives whole. `--fill zero` (black second
frame) is a measured-worse control, kept only for the ablation.

NOTE: transformers itself would silently duplicate a T=1 clip
(VJEPA2Embeddings.forward), so the expansion here is explicit on purpose — the
cache header then records exactly what was fed.

  python scripts/build_first_frame_cache.py --encoder vjepa2 --vjepa_grid 1 \
      --bank outputs/frame_bank.npz --out outputs/xpair_cache/vjepa2_first1dup
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.cache_io import CacheHeader, write_cache
from src.traj_data.encoder import (
    MIN_FRAMES,
    build_traj_encoder,
    default_model_id,
    encoder_format,
    expand_clip,
    parse_format,
)

# The bank holds one frame per (episode, variant); a real [f0, f1] pair would need
# the video path, so this builder only ever emits n_frames=1 records.
N_FRAMES = 1


def build_first_frame_records(images, episode, task_index, variant, encode,
                              min_frames: int, fill: str, keep_variants: bool = False):
    """images: (N,H,W,3) uint8 first frames; the other three are length-N arrays.
    `encode` maps (1,T,C,H,W) in [0,1] -> (1,L,D).

    Returns (seqs, records) sorted by (episode, variant) — the same order an
    unsharded build_xpair_cache produces, so readers cannot tell the two apart.
    """
    order = sorted(range(len(episode)),
                   key=lambda i: (int(episode[i]), int(variant[i])))
    seqs, records = [], []
    for i in order:
        var = int(variant[i])
        if not keep_variants and var != 0:
            continue
        frame = torch.from_numpy(np.asarray(images[i])).permute(2, 0, 1).float() / 255.0
        clip = expand_clip(frame.unsqueeze(0), min_frames, fill)      # (T,C,H,W)
        toks = encode(clip.unsqueeze(0))[0].cpu().numpy().astype(np.float16)
        seqs.append(toks)
        records.append({"episode": int(episode[i]), "variant": var,
                        "task_index": int(task_index[i])})
    if not seqs:
        raise ValueError("no records built — is the bank empty?")
    return seqs, records


def main(argv=None):  # pragma: no cover (GPU/weights)
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="outputs/frame_bank.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="vjepa2")          # dino | vjepa2
    p.add_argument("--encoder_model", default=None)
    p.add_argument("--vjepa_grid", type=int, default=1,
                   help="vjepa2: s×s tokens per tubelet (1 = full mean-pool, the "
                        "single-vector 'CLS analogue' this arm wants)")
    p.add_argument("--dino_grid", type=int, default=0,
                   help="dino: s×s patch tokens per frame (16 = ALL 256 patches raw)")
    p.add_argument("--dino_n_reg", type=int, default=0,
                   help="dino: register tokens (dinov2-with-registers only)")
    p.add_argument("--fill", default="dup", choices=["dup", "zero"],
                   help="how the single frame is expanded to the encoder minimum")
    p.add_argument("--keep_variants", action="store_true",
                   help="also encode the bank's sim-augmented variants (default: "
                        "originals only, matching the clean-grid protocol)")
    args = p.parse_args(argv)

    with np.load(args.bank, allow_pickle=False) as z:
        images, episode, task_index = z["images"], z["episode"], z["task_index"]
        # Banks built without sim-augmentations carry no `variant` (FrameBank
        # tolerates that too) — every row is then an original.
        variant = z["variant"] if "variant" in z.files else np.zeros(len(episode), int)
        task_texts = ({int(k): v for k, v in json.loads(str(z["task_texts_json"])).items()}
                      if "task_texts_json" in z.files else {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, encode = build_traj_encoder(args.encoder, args.encoder_model, device,
                                   vjepa_grid=args.vjepa_grid, dino_grid=args.dino_grid,
                                   n_reg=args.dino_n_reg)
    min_frames = MIN_FRAMES[args.encoder]
    seqs, records = build_first_frame_records(
        images, episode, task_index, variant, encode,
        min_frames, args.fill, keep_variants=args.keep_variants)

    # dino needs no expansion, so `fill` never runs and must NOT reach the tag:
    # otherwise --fill dup and --fill zero would label byte-identical caches
    # differently and make them mutually unloadable for no reason.
    fill_tag = args.fill if min_frames > N_FRAMES else ""
    fmt = encoder_format(args.encoder, args.vjepa_grid, time_select="first",
                         n_frames=N_FRAMES, fill=fill_tag, dino_grid=args.dino_grid,
                         n_reg=args.dino_n_reg)
    header = CacheHeader(
        encoder_id=args.encoder,
        format=fmt,
        d_enc=int(seqs[0].shape[-1]),
        aug_set="orig+bank" if args.keep_variants else "orig",
        num_records=len(records),
        time_select="first", n_frames=N_FRAMES, fill=fill_tag,
        # The whole (expanded) clip goes through a single forward, so the encode
        # window IS its length — this is what distinguishes these tokens from the
        # first tubelet of a chunk=32 full-clip build, which saw 32 frames.
        chunk=min_frames,
        encoder_model=args.encoder_model or default_model_id(args.encoder),
        tokens_per_unit=parse_format(fmt)["tokens_per_unit"])
    write_cache(args.out, seqs, records, header, task_texts)
    print(f"wrote {header.num_records} records ({sum(s.shape[0] for s in seqs)} tokens, "
          f"format={header.format}) to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
