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


class TrajCache:
    def __init__(self, out_dir: str):
        with open(os.path.join(out_dir, "index.json")) as fh:
            meta = json.load(fh)
        self.header = meta["header"]
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

    def assert_header_matches(self, **expected) -> None:
        for k, v in expected.items():
            got = self.header.get(k)
            assert got == v, f"cache header.{k}={got} != expected {v}"

    def read_row(self, row: int) -> torch.Tensor:
        r = self.records[row]
        a = np.asarray(self.tokens[r["offset"]:r["offset"] + r["length"]]).copy()
        return torch.from_numpy(a)

    def rows_of_task(self, task_index: int) -> list:
        return list(self._by_task.get(task_index, []))

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
