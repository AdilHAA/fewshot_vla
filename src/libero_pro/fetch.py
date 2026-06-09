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


def fetch_libero_pro(repo_id: str = LIBERO_PRO_REPO) -> Path:
    """Download the bddl/init files into the HF cache (idempotent); return root."""
    logger.info("Fetching LIBERO-Pro perturbation files from %s", repo_id)
    root = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=["bddl_files/**", "init_files/**"],
    )
    return Path(root)
