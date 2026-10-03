"""Paths for generated nanochat data and artifacts."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def get_base_dir():
    """Use this checkout's training/ unless NANOCHAT_BASE_DIR is set."""
    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR") or REPO_ROOT / "training").expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)
