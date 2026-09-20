"""On-disk ragged cache reader: per-record token reads + task_key lookup.

Task identity is the LIBERO bddl stem (`task_key` = `LiberoEnv.task`), NOT the
dataset's `task_index`: the merged HN dataset rebuilds task_index from instruction
TEXT, so it collapses the LIBERO-90 tasks that share a sentence across scenes (and
two texts across the two source datasets). Keying the conditioning pool by
task_index would therefore mix demos of different scenes into one "task".

Caches built before the registry carry only `task_index`; they still load, with
task_key = str(task_index) and src_len = 0 (genuinely unknown), so every consumer
sees one record shape.
"""
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
        self._ep_key: dict = {}
        self._ep_row: dict = {}
        for i, r in enumerate(self.records):
            if r.get("task_key") is None:                  # legacy: task_index only
                r["task_key"] = str(r["task_index"])
            r.setdefault("src_len", 0)
            ep, var = int(r["episode"]), int(r.get("variant", 0))
            self._by_task.setdefault(r["task_key"], []).append(i)
            self._ep_key.setdefault(ep, r["task_key"])
            self._ep_row.setdefault((ep, var), i)
        # task_texts lives at the TOP level of index.json (not in the header): the
        # trunk arm feeds these STRINGS through its own tokenizer. Keys are task_keys
        # (legacy caches: str(task_index)); json round-trips them as strings anyway.
        self.task_texts: dict = {str(k): v for k, v in meta.get("task_texts", {}).items()}
        # Text -> key is only the LEGACY eval path: twins share a sentence, so this
        # map is lossy by construction. `resolve_key` (exact bddl stem) is the real one.
        self._text_to_key = {norm_text(v): k for k, v in self.task_texts.items()}

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

    def rows_of_task(self, task_key: str, originals_only: bool = True) -> list:
        """Row indices of a task; by default only original recordings (variant==0),
        excluding sim-augmented variants from the conditioning pool."""
        rows = self._by_task.get(str(task_key), [])
        if originals_only:
            return [r for r in rows if self.records[r].get("variant", 0) == 0]
        return list(rows)

    def key_of_episode(self, episode: int) -> str:
        """task_key of a dataset episode. KeyError if the cache does not hold it —
        conditioning on the wrong task is worse than a crash, so there is no default."""
        return self._ep_key[int(episode)]

    def row_of_episode(self, episode: int, variant: int = 0) -> int:
        return self._ep_row[(int(episode), int(variant))]

    def task_keys(self) -> list:
        return sorted(self._by_task)

    def resolve_key(self, name: str):
        """LIBERO task name (bddl stem, with or without '.bddl') -> task_key.

        Exact match only: the stems of all 130 tasks are unique, and LIBERO-Pro's
        perturbed suites reuse the base stems verbatim, so a Pro rollout resolves to
        the task its demos were recorded for."""
        n = (name or "").strip()
        if n.endswith(".bddl"):
            n = n[: -len(".bddl")]
        return n if n in self._by_task else None

    def resolve_task(self, text: str):
        """LEGACY instruction -> task_key: exact normalized match, then fuzzy."""
        n = norm_text(text)
        if n in self._text_to_key:
            return self._text_to_key[n]
        hit = difflib.get_close_matches(n, list(self._text_to_key), n=1, cutoff=0.6)
        return self._text_to_key[hit[0]] if hit else None

    def nearest_task(self, text: str):
        """Best-effort fallback: the closest task text with no cutoff (for novel
        instructions that legitimately match nothing). None only on an empty cache."""
        if not self._text_to_key:
            return None
        hit = difflib.get_close_matches(norm_text(text), list(self._text_to_key),
                                        n=1, cutoff=0.0)
        return self._text_to_key[hit[0]] if hit else None
