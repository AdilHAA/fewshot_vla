"""Helpers for the `lerobot/libero` dataset.

`lerobot/libero`'s `meta/episodes` parquet ships a wrong `data/file_index`
(episodes 810-1259 map to data files that actually hold episodes 90-139; the
video file_index is correct). Episode-based file selection therefore loads the
wrong rows. Pre-downloading every data parquet lets lerobot's loader glob them
all and filter by the `episode_index` column instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

logger = logging.getLogger(__name__)

DEFAULT_LIBERO_REPO = "lerobot/libero"
LIBERO_REVISION = "v3.0"


def prefetch_all_data_parquets(
    repo_id: str = DEFAULT_LIBERO_REPO, revision: str = LIBERO_REVISION
) -> Path:
    """Download all data parquets (~38 MB, idempotent) and return the snapshot root."""
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            allow_patterns=["data/**/*.parquet"],
        )
    )
