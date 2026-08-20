"""On-disk ragged cache reader: per-record token reads + base-task lookup."""
from __future__ import annotations

import difflib
import json
import os
import re

import numpy as np
import torch


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


from src.traj_data.cache_io import HEADER_DEFAULTS

# Header keys introduced 2026-08-02. Caches written before that lack them, so the
# knowable ones are back-filled with what those builds implicitly had — otherwise
# every pre-existing cache would fail `assert_header_matches` on `None != "all"`.
# `chunk`/`encoder_model` are back-filled with the UNRECORDED sentinels instead:
# their real values cannot be recovered, so pinning them against an old cache is
# meant to fail rather than silently "match".
_UNRECORDED = {"chunk": 0, "encoder_model": ""}


class TrajCache:
    def __init__(self, out_dir: str):
        with open(os.path.join(out_dir, "index.json")) as fh:
            meta = json.load(fh)
        self.header = meta["header"]
        for k, v in {**HEADER_DEFAULTS, **_UNRECORDED}.items():
            self.header.setdefault(k, v)
        # tokens_per_unit is DERIVABLE from the format tag, so no cache ever needs a
        # rebuild for it — unlike chunk/encoder_model, which are genuinely lost.
        if not self.header.get("tokens_per_unit"):
            from src.traj_data.encoder import parse_format
            self.header["tokens_per_unit"] = parse_format(self.header["format"])["tokens_per_unit"]
        self.records = meta["records"]
        self._d = int(self.header["d_enc"])
        total = sum(r["length"] for r in self.records)
        path = os.path.join(out_dir, "tokens.mmap")
        expected = total * self._d * 2                     # fp16 = 2 bytes
        actual = os.path.getsize(path)
        assert actual == expected, f"tokens.mmap size {actual} != expected {expected}"
        self.tokens = np.memmap(path, dtype=np.float16, mode="r", shape=(total, self._d))
        self._by_task: dict = {}
        for i, r in enumerate(self.records):
            self._by_task.setdefault(r["task_index"], []).append(i)
        self._text_to_task = {norm_text(v): int(k)
                              for k, v in meta.get("task_texts", {}).items()}
        # task_texts lives at the TOP level of index.json (not in the header): the
        # trunk arm feeds these STRINGS through its own tokenizer, so expose them
        # keyed by both str and int (json round-trips keys as strings).
        self.task_texts: dict = {}
        for k, v in meta.get("task_texts", {}).items():
            self.task_texts[str(k)] = v
            try:
                self.task_texts[int(k)] = v
            except (TypeError, ValueError):
                pass

    def assert_header_matches(self, **expected) -> None:
        for k, v in expected.items():
            got = self.header.get(k)
            if got != v:
                raise ValueError(f"cache header.{k}={got} != expected {v} — "
                                 f"rebuild the cache for this encoder config")

    def read_row(self, row: int) -> torch.Tensor:
        r = self.records[row]
        a = np.asarray(self.tokens[r["offset"]:r["offset"] + r["length"]]).copy()
        return torch.from_numpy(a)

    def rows_of_task(self, task_index: int, originals_only: bool = True) -> list:
        """Row indices of a task; by default only original recordings (variant==0),
        excluding sim-augmented variants from the conditioning pool."""
        rows = self._by_task.get(task_index, [])
        if originals_only:
            return [r for r in rows if self.records[r].get("variant", 0) == 0]
        return list(rows)

    def resolve_task(self, text: str):
        """Instruction -> task_index: exact normalized match, then fuzzy (paraphrases)."""
        n = norm_text(text)
        if n in self._text_to_task:
            return self._text_to_task[n]
        hit = difflib.get_close_matches(n, list(self._text_to_task), n=1, cutoff=0.6)
        return self._text_to_task[hit[0]] if hit else None

    def nearest_task(self, text: str):
        """Best-effort fallback: the closest task text with no cutoff (for novel
        instructions that legitimately match nothing). None only on an empty cache."""
        if not self._text_to_task:
            return None
        hit = difflib.get_close_matches(norm_text(text), list(self._text_to_task),
                                        n=1, cutoff=0.0)
        return self._text_to_task[hit[0]] if hit else None
