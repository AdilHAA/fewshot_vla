"""On-disk ragged cache layout (write side): tokens.mmap + index.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CacheHeader:
    encoder_id: str        # "dino" | "vjepa2"
    format: str            # "cls" | "tubelet_grid{s}" | "<base>@first{n}{fill}"
    d_enc: int
    aug_set: str           # provenance tag, e.g. "orig+sim-recolor+cam-jitter"
    num_records: int
    # --- provenance (added 2026-08-02); defaults reproduce every earlier cache ----
    # Without these, caches that differ in ways the TOKENS cannot show are
    # indistinguishable: dinov2-base vs dinov2-with-registers-base used to share a
    # header byte-for-byte, and so did the chunk=32 and chunk=64 V-JEPA2 builds.
    time_select: str = "all"   # "all" | "first" — temporal scope of each record
    n_frames: int = 0          # frames kept per record when time_select="first"
    fill: str = ""             # "" | "dup" | "zero" — how a short clip was padded
    # Tokens per unit (frame for dino, tubelet for vjepa2). Tokens are unit-major,
    # so the temporal index of slot i is i // tokens_per_unit — without this the
    # positional code would give the N tokens of ONE frame N different times.
    # 0 = derive from `format` (every legacy tag is self-describing).
    tokens_per_unit: int = 0
    # 0 means UNRECORDED, not "one forward": every builder now writes the real
    # window, so 0 only ever appears on a cache built before these fields existed.
    # Pinning `chunk` therefore (correctly) refuses to validate such a cache.
    chunk: int = 0
    encoder_model: str = ""    # HF model id; `encoder_id` does NOT identify weights


# Header keys that must agree for two shards to belong to the same cache. `d_enc`
# and `aug_set` are here (not just in the policy check) because merging shards that
# disagree writes irrecoverable garbage into one mmap.
SHARD_KEYS = ("encoder_id", "format", "d_enc", "aug_set", "time_select",
              "n_frames", "fill", "chunk", "encoder_model", "tokens_per_unit")

# Values the pre-2026-08-02 builders implicitly had, used to read old caches and to
# compare an old shard with a freshly rebuilt one. `chunk`/`encoder_model` are NOT
# here: they are genuinely unknown for those caches, and inventing a value would
# turn "unrecorded" into a false provenance claim.
HEADER_DEFAULTS = {"time_select": "all", "n_frames": 0, "fill": ""}


class CacheWriter:
    """Append-only streaming writer: keeps ZERO token bytes in RAM.

    `write_cache` needs every sequence in memory at once to size the memmap, which
    is fine for a 0.4 GB `cls` build and impossible for a raw-patch one (257
    tokens/frame over 273k frames = 108 GB). Appending to a plain file produces a
    byte-identical `tokens.mmap` — the reader derives its shape from index.json —
    so nothing downstream can tell the two writers apart.

        w = CacheWriter(out_dir, d_enc)
        for toks, rec in ...:  w.add(toks, rec)
        w.close(header, task_texts)
    """

    def __init__(self, out_dir: str, d_enc: int, bufsize: int = 1 << 24):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.d_enc = out_dir, int(d_enc)
        self._fh = open(os.path.join(out_dir, "tokens.mmap"), "wb", buffering=bufsize)
        self.records: list[dict] = []
        self._off = 0

    def add(self, seq, rec: dict) -> None:
        if seq.dtype != np.float16 or seq.shape[1] != self.d_enc:
            raise ValueError(f"expected (L,{self.d_enc}) fp16, got {seq.shape} {seq.dtype}")
        self._fh.write(np.ascontiguousarray(seq).tobytes())
        n = int(seq.shape[0])
        self.records.append({**rec, "offset": self._off, "length": n})
        self._off += n

    def close(self, header: CacheHeader, task_texts: dict) -> int:
        self._fh.close()
        header.num_records = len(self.records)
        header.d_enc = self.d_enc
        with open(os.path.join(self.out_dir, "index.json"), "w") as fh:
            json.dump({"header": asdict(header), "records": self.records,
                       "task_texts": {str(k): v for k, v in task_texts.items()}}, fh)
        return self._off


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
