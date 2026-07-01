"""On-disk cache reader: clip reads + brute-force cosine retrieval + provenance."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from .retrieval import cosine_topk


@dataclass
class RetrievalResult:
    rows: list
    sims: list
    kill_switch: bool
    provenance: list


class TrajCache:
    def __init__(self, out_dir: str):
        with open(os.path.join(out_dir, "index.json")) as fh:
            meta = json.load(fh)
        self.header = meta["header"]
        self.records = meta["records"]
        self.keys = torch.load(os.path.join(out_dir, "keys.pt"))
        h = self.header
        self._tok = h.get("n_tokens") or (h["num_frames"] * h["ntok"])
        self._d = h["d_enc"]
        n = h["n_clips"]
        path = os.path.join(out_dir, "clips.mmap")
        # fp16 = 2 bytes; validate before mmap to catch truncated writes early.
        expected = n * self._tok * self._d * 2
        actual = os.path.getsize(path)
        assert actual == expected, f"clips.mmap size {actual} != expected {expected}"
        self.clips = np.memmap(path, dtype=np.float16, mode="r",
                               shape=(n, self._tok, self._d))
        self._row = {(r["episode"], r["variant"]): r["row"] for r in self.records}
        self._task_eps: dict = {}
        for r in self.records:
            self._task_eps.setdefault(r["task_index"], set()).add(r["episode"])
        self._row_task = {r["row"]: r["task_index"] for r in self.records}

    def assert_header_matches(self, **expected) -> None:
        for k, v in expected.items():
            got = self.header.get(k)
            assert got == v, f"cache header.{k}={got} != expected {v}"

    def read(self, episode: int, variant: int) -> torch.Tensor:
        row = self._row.get((episode, variant))
        if row is None:
            raise KeyError(f"(episode={episode}, variant={variant}) not in cache")
        return torch.from_numpy(np.asarray(self.clips[row]).copy())

    def read_rows(self, rows: list) -> torch.Tensor:
        return torch.stack([torch.from_numpy(np.asarray(self.clips[r]).copy()) for r in rows])

    def episodes_of_task(self, task_index: int) -> list:
        return sorted(self._task_eps.get(task_index, set()))

    def retrieve(self, query: torch.Tensor, k: int, tau: float,
                 query_task_index: int) -> RetrievalResult:
        idx, sims, kill = cosine_topk(query, self.keys, k=k, tau=tau)
        rows = idx.tolist()
        prov = ["same_task" if self._row_task[r] == query_task_index
                else "different_task" for r in rows]
        return RetrievalResult(rows=rows, sims=sims.tolist(), kill_switch=kill,
                               provenance=prov)
