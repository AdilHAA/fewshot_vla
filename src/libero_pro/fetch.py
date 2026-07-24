"""Fetch the pre-generated LIBERO-Pro perturbation files from the HF Hub.

``zhouxueyang/LIBERO-Pro`` ships only the bddl/init task files; they run on the
native LIBERO simulator lerobot already wraps (no upstream LIBERO-PRO runtime).
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

LIBERO_PRO_REPO = "zhouxueyang/LIBERO-Pro"
# Pinned to the last revision before PR #4 ("seven-case robustness set"), which
# added new-scheme axis dirs (e.g. 01_visual_noise_glare) our 4-axis grid doesn't
# use. Unpinned `main` would otherwise silently change the registered suite set.
LIBERO_PRO_REVISION = "bcaa82b8"


def fetch_libero_pro(repo_id: str = LIBERO_PRO_REPO,
                     revision: str = LIBERO_PRO_REVISION) -> Path:
    """Download the bddl/init files into the HF cache (idempotent); return root."""
    logger.info("Fetching LIBERO-Pro perturbation files from %s@%s", repo_id, revision)
    root = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["bddl_files/**", "init_files/**"],
    )
    return Path(root)
