"""Download the 13 Tier P (permitted) Brain Treebank recordings into data_dev/.

Runs ON the server, sequentially, one archive on disk at a time:

    nice -n 19 ionice -c 3 .venv/bin/python code/server/download_dev.py [--dry-run]

Everything else Tier P needs (electrode labels, localization, timings, metadata,
transcripts, corrupted_elec.json) already exists for all 26 sessions in the
reproduction's data/ and is symlinked into data_dev/ by server/setup.sh, so only
the recordings are fetched. Session 6_0 and the partial sessions are not Tier P
(their films are shown in evaluation sessions) and are never downloaded.

Before each archive the script checks that free space on / would stay above
--min-free-gib (default 350) with the archive and its extraction on disk, and
stops (exit 3) if not. Archives are deleted after extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BASE = "https://braintreebank.dev/data/"
TIER_P = [(1, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 2), (4, 2), (5, 0), (6, 1), (6, 4), (8, 0), (9, 0)]
GIB = 1024 ** 3
EXPANSION = 1.03  # h5 is already gzip-compressed inside; measured 1.02 on the 12 evaluation archives


def name(subject: int, trial: int) -> str:
    return f"sub_{subject}_trial{trial:03d}.h5"


def remote_size(url: str) -> int:
    # curl, not urllib: the uv-managed interpreter on this NixOS host has no CA bundle configured.
    head = subprocess.run(["curl", "--head", "--silent", "--show-error", "--fail", "--location",
                           "--connect-timeout", "15", "--retry", "3", url],
                          check=True, capture_output=True, text=True).stdout
    sizes = [line.split(":", 1)[1] for line in head.splitlines() if line.lower().startswith("content-length:")]
    return int(sizes[-1])


def free_gib(path: Path) -> float:
    """Free space of the filesystem that holds ``path`` (the data directory being written)."""
    return shutil.disk_usage(path).free / GIB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("NPEXP_ROOT", "~/runs/neuroprobe-experiments")).expanduser())
    parser.add_argument("--min-free-gib", type=float, default=350.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = args.root / "data_dev"
    staging = data / ".downloads"
    receipts = data / ".receipts"
    for folder in (staging, receipts):
        folder.mkdir(parents=True, exist_ok=True)

    todo, total = [], 0
    for subject, trial in TIER_P:
        target = data / name(subject, trial)
        receipt = receipts / (name(subject, trial) + ".json")
        if target.exists() and receipt.exists():
            print(f"have    {target.name}  {target.stat().st_size / GIB:6.2f} GiB")
            continue
        url = f"{BASE}subject_data/sub_{subject}/trial{trial:03d}/{name(subject, trial)}.zip"
        size = remote_size(url)
        todo.append((subject, trial, url, size))
        total += size
        print(f"missing {target.name}  archive {size / GIB:6.2f} GiB")
    peak = max((size for *_, size in todo), default=0) * (1 + EXPANSION)
    print(f"to fetch: {len(todo)} archives, {total / GIB:.1f} GiB compressed, "
          f"about {total * EXPANSION / GIB:.1f} GiB kept after extraction; "
          f"largest transient need {peak / GIB:.1f} GiB; free under {data} now {free_gib(data):.1f} GiB; "
          f"floor {args.min_free_gib:.0f} GiB")
    if args.dry_run or not todo:
        return 0

    for subject, trial, url, size in todo:
        target = data / name(subject, trial)
        need = size * (1 + EXPANSION) / GIB
        if free_gib(data) - need < args.min_free_gib:
            print(f"STOP before {target.name}: free {free_gib(data):.1f} GiB minus {need:.1f} GiB "
                  f"would fall below {args.min_free_gib:.0f} GiB", flush=True)
            return 3
        archive = staging / (target.name + ".zip")
        started = time.time()
        subprocess.run(["curl", "--retry", "5", "--retry-all-errors", "--connect-timeout", "15", "--fail",
                        "--silent", "--show-error", "--location", "--continue-at", "-",
                        "-o", str(archive), url], check=True)
        if archive.stat().st_size != size:
            raise RuntimeError(f"{archive}: {archive.stat().st_size} bytes, expected {size}")
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.namelist() if m.endswith(target.name)]
            if len(members) != 1:
                raise RuntimeError(f"{archive}: expected one {target.name}, found {bundle.namelist()}")
            extracted = Path(bundle.extract(members[0], staging))
        archive.unlink()
        digest = sha256(extracted)
        os.replace(extracted, target)
        (receipts / (target.name + ".json")).write_text(json.dumps(
            {"url": url, "archive_bytes": size, "h5_bytes": target.stat().st_size, "h5_sha256": digest,
             "seconds": round(time.time() - started, 1)}))
        print(f"done    {target.name}  {target.stat().st_size / GIB:6.2f} GiB  "
              f"{time.time() - started:6.0f} s  free {free_gib(data):.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
