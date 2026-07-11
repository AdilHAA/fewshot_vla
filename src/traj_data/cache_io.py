"""On-disk ragged cache layout (write side): tokens.mmap + index.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CacheHeader:
    encoder_id: str        # "dino" | "vjepa2"
    format: str            # "cls" | "tubelet_meanpool"
    d_enc: int
    aug_set: str           # provenance tag, e.g. "orig+sim-recolor+cam-jitter"
    num_records: int


def write_cache(out_dir: str, token_seqs: list, records: list[dict],
                header: CacheHeader, task_texts: dict) -> None:
    """token_seqs[i] is a (L_i, d_enc) fp16 array for records[i] ({episode, variant,
    task_index}); offset/length are computed here. task_texts maps task_index -> the
    task instruction (eval-time text fallback)."""
    os.makedirs(out_dir, exist_ok=True)
    assert len(token_seqs) == len(records) == header.num_records
    total = sum(int(s.shape[0]) for s in token_seqs)
    mm = np.memmap(os.path.join(out_dir, "tokens.mmap"), dtype=np.float16,
                   mode="w+", shape=(total, header.d_enc))
    off, out_records = 0, []
    for seq, rec in zip(token_seqs, records):
        assert seq.dtype == np.float16 and seq.shape[1] == header.d_enc
        n = int(seq.shape[0])
        mm[off:off + n] = seq
        out_records.append({**rec, "offset": off, "length": n})
        off += n
    mm.flush()
    del mm
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"header": asdict(header), "records": out_records,
                   "task_texts": {str(k): v for k, v in task_texts.items()}}, fh)
