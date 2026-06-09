#!/usr/bin/env python3
"""Post-install patch for third-party lerobot.

lerobot 0.5.1 ships GR00TN15Config decorated with bare @dataclass while
declaring fields(init=False) without defaults. Under transformers>=5 /
Python 3.12 this raises at import time:

    TypeError: non-default argument 'backbone_cfg' follows default argument

The class defines its own __init__, so the dataclass-generated one is
not needed. Switching to @dataclass(init=False) sidesteps the field
ordering check entirely.

Idempotent: re-running does nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import lerobot
except ImportError:
    sys.exit("lerobot not installed; run pip install first")

target = Path(lerobot.__file__).parent / "policies" / "groot" / "groot_n1.py"
if not target.exists():
    sys.exit(f"expected file not found: {target}")

src = target.read_text()
patched_marker = "@dataclass(init=False)\nclass GR00TN15Config(PretrainedConfig):"
original_marker = "@dataclass\nclass GR00TN15Config(PretrainedConfig):"

if patched_marker in src:
    print(f"[patch_lerobot] already patched: {target}")
    sys.exit(0)
if original_marker not in src:
    sys.exit(
        f"[patch_lerobot] expected marker not found in {target}. "
        "lerobot version may have changed; review groot_n1.py manually."
    )

target.write_text(src.replace(original_marker, patched_marker, 1))
print(f"[patch_lerobot] applied @dataclass(init=False) to {target}")
