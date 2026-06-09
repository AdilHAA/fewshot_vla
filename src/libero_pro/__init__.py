"""LIBERO-Pro integration: register the perturbed-task suites so lerobot's
``LiberoEnv`` can target them. Data-only — consumes the BDDL/init files from
``zhouxueyang/LIBERO-Pro``; importing the package fetches + registers them.

Axes (suffix → axis): _lan → Semantic, _object → Object, _swap → Position,
_task → Task. NOTE: the Object axis needs object assets absent from stock
LIBERO and currently fails to build; the other three work.
"""

from __future__ import annotations

from .fetch import fetch_libero_pro
from .register import (
    BASE_SUITES,
    PERTURBATIONS,
    ensure_registered,
    perturbed_suite_names,
)

# Register on import so the eval shim needs no explicit call.
ensure_registered()

__all__ = [
    "fetch_libero_pro",
    "ensure_registered",
    "perturbed_suite_names",
    "BASE_SUITES",
    "PERTURBATIONS",
]
