"""Pin the normalizer to the frozen base's own statistics.

lerobot 0.5.1's `lerobot_train` builds the pre/post-processors from the TRAINING
dataset's `meta.stats` — always, even when a pretrained policy is given. That is
right when the policy is trained (the base finetune itself ran that way), and wrong
for a frozen `base_smolvla_path`: the base was trained under its dataset's
mean/std, and feeding it states normalised with another dataset's mean/std (and
un-normalising its actions the same way) changes what it computes. Measured on
hn_scratch (2026-09-21): the base's loss on ITS OWN episodes went 0.05 -> 0.37 just
from the merged libero_all stats. The HN then trains against a corrupted base, and
eval inherits the same stats from the checkpoint.

The fix reads the base checkpoint's saved normalizer
(`policy_preprocessor.json` -> normalizer step -> its safetensors, keys like
`observation.state.mean`) and replaces the dataset stats for the policy's
non-visual features with it. Lerobot-free so it is testable headless; the trainer
glue lives in train_hyper_lora.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

PREPROCESSOR_JSON = "policy_preprocessor.json"


def _resolve(base_path: str, filename: str) -> Path:
    local = Path(base_path) / filename
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(base_path, filename))


def load_base_normalizer_stats(base_path: str) -> dict[str, dict[str, np.ndarray]]:
    """{feature_key: {stat_name: array}} from the checkpoint's normalizer step.
    The safetensors keys are '<feature>.<stat>' with the stat being the last dotted
    component (feature keys themselves contain dots: 'observation.state')."""
    pre = json.loads(_resolve(base_path, PREPROCESSOR_JSON).read_text())
    steps = [s for s in pre["steps"] if s.get("registry_name") == "normalizer_processor"]
    if len(steps) != 1 or not steps[0].get("state_file"):
        raise ValueError(f"{base_path}: expected exactly one normalizer_processor step "
                         f"with a state_file in {PREPROCESSOR_JSON}, got {steps}")
    tensors = load_file(str(_resolve(base_path, steps[0]["state_file"])))
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, v in tensors.items():
        feature, stat = k.rsplit(".", 1)
        out.setdefault(feature, {})[stat] = np.asarray(v)
    return out


def pin_base_stats(dataset_stats: dict, base_stats: dict, keys: list[str]) -> tuple[dict, list[str]]:
    """Copy of `dataset_stats` with every key in `keys` REPLACED by the base's stats
    (replace, not merge: a leftover 'min'/'max' from the dataset would describe a
    different distribution). A key absent from the base normalizer is an error — the
    base cannot have consumed a feature it has no stats for."""
    out = copy.deepcopy(dataset_stats)
    pinned: list[str] = []
    for key in keys:
        if key not in base_stats:
            raise KeyError(f"feature {key!r} has no stats in the base normalizer "
                           f"(has: {sorted(base_stats)})")
        out[key] = {stat: np.array(v, copy=True) for stat, v in base_stats[key].items()}
        pinned.append(key)
    return out, pinned
