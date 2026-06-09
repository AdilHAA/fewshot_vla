"""Derive SmolVLA config overrides from a base checkpoint's config.json.

`smolvla_libero` was trained with non-default fields (expert_width_multiplier=0.5
→ expert hidden 480 vs the default 720, num_vlm_layers=0, ...). Building our
config from dataclass defaults then loading the base state_dict would mismatch,
so we copy the base's scalar architecture fields onto the CLI as `--policy.*`
overrides, skipping runtime/training knobs and anything the user already passed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Not architecture (don't affect state_dict shapes): policy-type discriminator,
# runtime knobs, publishing metadata, optimizer/scheduler schedule.
_DENYLIST: frozenset[str] = frozenset({
    "type", "device", "use_amp",
    "push_to_hub", "private", "repo_id", "license", "tags",
    "optimizer_betas", "optimizer_eps", "optimizer_grad_clip_norm",
    "optimizer_lr", "optimizer_weight_decay",
    "scheduler_decay_lr", "scheduler_decay_steps", "scheduler_warmup_steps",
})


def _fmt(v: object) -> str | None:
    """Format a scalar for the CLI; None skips the field (lists/dicts/null are
    dataset-derived or already default, and don't affect state_dict shapes)."""
    if isinstance(v, bool):  # before int — bool is an int subclass
        return "true" if v else "false"
    if isinstance(v, (int, float, str)):
        return str(v)
    return None


def overrides_from_config_dict(cfg: dict, argv: list[str]) -> list[str]:
    already = {a.split("=", 1)[0] for a in argv if a.startswith("--policy.") and "=" in a}
    out: list[str] = []
    for key in sorted(cfg):
        flag = f"--policy.{key}"
        if key in _DENYLIST or flag in already:
            continue
        s = _fmt(cfg[key])
        if s is not None:
            out.append(f"{flag}={s}")
    return out


def _load_base_config_dict(base_path: str) -> dict | None:
    local = Path(base_path) / "config.json"
    try:
        if local.is_file():
            return json.loads(local.read_text())
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(base_path, "config.json")).read_text())
    except Exception as e:  # noqa: BLE001 — never fatal; fall back to no overrides
        logger.warning("Could not load base config.json from %r: %r", base_path, e)
        return None


def base_config_overrides(base_path: str, argv: list[str]) -> list[str]:
    cfg = _load_base_config_dict(base_path)
    return overrides_from_config_dict(cfg, argv) if cfg else []
