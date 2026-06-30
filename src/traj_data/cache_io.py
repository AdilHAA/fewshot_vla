"""On-disk cache layout (write side): clips.mmap + keys.pt + index.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class CacheHeader:
    encoder_id: str
    num_frames: int
    stride: int
    window: str
    jitter: str
    d_enc: int
    ntok: int
    sentence_encoder_id: str
    beta_t: float
    beta_f: float
    d_text: int
    d_frame: int
    n_clips: int


def write_cache(out_dir: str, clips: np.ndarray, keys: torch.Tensor,
                records: list[dict], header: CacheHeader) -> None:
    os.makedirs(out_dir, exist_ok=True)
    assert clips.dtype == np.float16, f"clips must be fp16, got {clips.dtype}"
    assert clips.shape[0] == header.n_clips, "n_clips must match clips.shape[0]"
    mm = np.memmap(os.path.join(out_dir, "clips.mmap"), dtype=np.float16,
                   mode="w+", shape=clips.shape)
    mm[:] = clips[:]
    mm.flush()
    del mm
    torch.save(keys.to(torch.float16).contiguous(), os.path.join(out_dir, "keys.pt"))
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"header": asdict(header), "records": records}, fh)
