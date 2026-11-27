"""Shared paths and run identity for the classical and ensemble tracks.

Server layout (``$NPEXP_ROOT``, default ``~/runs/neuroprobe-experiments``)::

    code/      rsync copy of this folder (server/sync.sh writes code/COMMIT.json)
    data_dev/  merged ROOT_DIR_BRAINTREEBANK: symlinks to the reproduction's data
               plus the 13 Tier P recordings (server/download_dev.py)
    work/      feature and kernel caches (deletable)
    results/   preds/, ledger.csv, test_access.log, LOCK.json, frozen/
    logs/
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent


def root() -> Path:
    return Path(os.environ.get("NPEXP_ROOT", "~/runs/neuroprobe-experiments")).expanduser()


def results_root() -> Path:
    return root() / "results"


def work_root() -> Path:
    return root() / "work"


def data_root() -> Path:
    """Directory used as ROOT_DIR_BRAINTREEBANK (honours an explicit override)."""
    return Path(os.environ.get("ROOT_DIR_BRAINTREEBANK", root() / "data_dev")).expanduser()


def commit() -> dict:
    """Identity of the code that is running: ``{"commit", "dirty", "tree_sha256"}``.

    The server copy has no git metadata, so server/sync.sh records the commit in
    ``COMMIT.json`` next to the code. A git checkout is the fallback.
    """
    stamp = CODE_DIR / "COMMIT.json"
    if stamp.exists():
        return json.loads(stamp.read_text())
    try:
        head = subprocess.run(["git", "-C", str(CODE_DIR), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(CODE_DIR), "status", "--porcelain", "--", "."],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty, "tree_sha256": ""}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True, "tree_sha256": ""}


def config_hash(config: dict) -> str:
    """Stable 16-hex digest of a JSON-serializable configuration."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
