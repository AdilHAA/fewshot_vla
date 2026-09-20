"""Eval-time task-key resolution, split out of the policy so it is testable headless.

`task_index` is NOT a task identity on the merged LIBERO dataset (it is rebuilt from
the instruction TEXT, so twin tasks living in different scenes collapse into one
index), and the instruction alone is ambiguous for the same reason. The identity is
the LIBERO bddl stem, which `eval_hyper_lora.py` forwards from `env.call('task')` as
`batch['subtask']`. The old text-matching path stays reachable behind
HN_TASK_BY_TEXT=1 for evals launched through stock lerobot-eval, which sets no
'subtask' — conditioning then silently picks demos of a twin task, hence the opt-in.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TEXT_KEYS = ("task", "instruction", "language_instruction", "prompt")


def _first_str(batch, key: str) -> str:
    """First entry of a per-env list (all envs of a rollout run the same task)."""
    v = batch.get(key) if isinstance(batch, dict) else None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return v.strip() if isinstance(v, str) else ""


def text_fallback_enabled() -> bool:
    return os.environ.get("HN_TASK_BY_TEXT") == "1"


def resolve_eval_key(batch, cache, allow_text: bool | None = None, decode_fn=None) -> str:
    """batch['subtask'] (a bddl stem per env) -> the conditioning cache's task key.

    `allow_text=None` reads HN_TASK_BY_TEXT; `decode_fn` is the last-resort
    instruction source of the legacy path (the policy decodes its language tokens).
    """
    if allow_text is None:
        allow_text = text_fallback_enabled()
    stem = _first_str(batch, "subtask")
    if stem:
        key = cache.resolve_key(stem)
        if key is not None:
            return key
        if not allow_text:
            raise RuntimeError(
                f"eval task key {stem!r} is in no conditioning cache task "
                f"({len(cache.task_keys())} keys) — the cache was built on a dataset "
                "that does not contain this task (a task_index-keyed legacy cache "
                "needs HN_TASK_BY_TEXT=1)")
        logger.warning("[TRAJ] task key %r not in the cache; HN_TASK_BY_TEXT=1 -> "
                       "matching the instruction text instead", stem)
    elif not allow_text:
        raise RuntimeError(
            "no 'subtask' in the eval batch, so the task key (LIBERO bddl stem) that "
            "selects the demo context is unknown. Run the eval through "
            "eval_hyper_lora.py, which forwards env.call('task') as batch['subtask']; "
            "HN_TASK_BY_TEXT=1 falls back to matching the instruction text, which is "
            "ambiguous across scenes.")

    text = ""
    for k in _TEXT_KEYS:
        text = _first_str(batch, k)
        if text:
            break
    if not text and decode_fn is not None:
        text = decode_fn() or ""
    key = cache.resolve_task(text)
    if key is None:
        # Novel-instruction suites (the LIBERO-Pro _task axis) legitimately miss the
        # cutoff — condition on the nearest known task instead of crashing.
        key = cache.nearest_task(text)
        if key is not None:
            logger.warning("[TRAJ] instruction %r matched no cached task; falling back "
                           "to nearest task %s", text, key)
    if key is None:
        keys = sorted(batch.keys()) if isinstance(batch, dict) else type(batch).__name__
        raise RuntimeError(f"cannot resolve eval task; instruction={text!r}, "
                           f"batch keys={keys}")
    return key
