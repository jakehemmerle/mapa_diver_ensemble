"""Append-only run ledger (``<root>/ledger.csv``) and the identifiers every row carries.

Every run of either track appends here, including failed and stopped arms. Rows are
chained: ``prev`` is a digest of the file as it was before the row was appended, so an
edited or deleted row is detectable with :func:`verify`. There is no update or delete.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = ("ts", "prev", "commit", "config_hash", "tier", "regime", "phase", "kind", "arm",
           "baseline", "candidate", "score", "baseline_score", "delta", "n_subjects",
           "subjects_up", "p_signflip", "ci_lo", "ci_hi", "decision", "extra")
REQUIRED = ("tier", "regime", "phase", "kind", "arm")
KINDS = ("comparison", "control", "member", "info")
_CODE_DIR = Path(__file__).resolve().parents[1]


class LedgerError(RuntimeError):
    pass


class CandidateConflict(LedgerError):
    """The ledger shows that this regime's single Tier V or Tier T pass is not ours to run."""


def root_dir() -> Path:
    return Path(os.environ.get("NPEXP_ROOT", Path.home() / "runs/neuroprobe-experiments"))


def results_root() -> Path:
    return root_dir() / "results"


def commit() -> str:
    """Commit of the code that is running: ``NPEXP_COMMIT``, then ``COMMIT.json`` beside
    the packages (written by the sync helper), then git. ``unknown`` if none is available."""
    value = os.environ.get("NPEXP_COMMIT", "").strip()
    if value:
        return value
    marker = _CODE_DIR / "COMMIT.json"
    if marker.is_file():
        info = json.loads(marker.read_text())
        return str(info["commit"]) + ("+dirty" if info.get("dirty") else "")
    try:
        head = subprocess.run(["git", "-C", str(_CODE_DIR), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_CODE_DIR), "status", "--porcelain", "--", "."],
                               check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_hash(config) -> str:
    """Stable 16-hex digest of a JSON-serializable configuration."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, default=str)
    return str(value).replace("\r", " ").replace("\n", " ")


def append(row: dict, root: Path) -> None:
    """Append one row. Unknown keys are kept in the ``extra`` JSON column."""
    missing = [key for key in REQUIRED if not row.get(key)]
    if missing:
        raise LedgerError(f"ledger row lacks {missing}")
    if row["kind"] not in KINDS:
        raise LedgerError(f"kind must be one of {KINDS}, got {row['kind']!r}")
    if "prev" in row or "ts" in row:
        raise LedgerError("ts and prev are assigned by the ledger")
    extra = dict(row.get("extra") or {})
    extra.update({k: v for k, v in row.items() if k not in COLUMNS})
    record = {k: row.get(k) for k in COLUMNS}
    record["extra"] = extra or None
    record["commit"] = row.get("commit") or commit()
    record["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / "ledger.csv", os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        before = b""
        while chunk := os.read(fd, 1 << 20):
            before += chunk
        record["prev"] = _digest(before)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        if not before:
            writer.writerow(COLUMNS)
        writer.writerow([_cell(record[k]) for k in COLUMNS])
        os.write(fd, buffer.getvalue().encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def read(root: Path) -> list[dict]:
    file = Path(root) / "ledger.csv"
    if not file.is_file():
        return []
    with file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path) -> None:
    """Raise if any row's ``prev`` does not match the bytes that precede it."""
    file = Path(root) / "ledger.csv"
    if not file.is_file():
        return
    data = file.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines or lines[0].decode().strip().split(",") != list(COLUMNS):
        raise LedgerError("ledger header is missing or altered")
    prev_index = COLUMNS.index("prev")
    seen = b""
    for number, line in enumerate(lines):
        if number:
            fields = next(csv.reader([line.decode()]))
            expected = _digest(b"" if number == 1 else seen)
            if len(fields) != len(COLUMNS) or fields[prev_index] != expected:
                raise LedgerError(f"ledger row {number} does not chain to the rows before it")
        seen += line


def guard_candidate(root: Path, tier: str, regime: str, candidate: str) -> None:
    """Enforce one frozen candidate, one Tier V pass and one Tier T pass per regime.

    Tier V: refuse if the ledger holds any Tier V row of this regime for a different
    candidate, or a Tier V comparison for this one (the pass has been used).
    Tier T: additionally require this candidate's Tier V comparison to read ``Go`` and
    refuse if a Tier T comparison already exists for the regime.
    """
    if tier not in ("V", "T"):
        return
    rows = [r for r in read(root) if r["regime"] == regime and r["candidate"]]
    veto = [r for r in rows if r["tier"] == "V"]
    others = sorted({r["candidate"] for r in veto} - {candidate})
    if others:
        raise CandidateConflict(
            f"Tier V in regime {regime} already holds rows for candidate(s) {others}; "
            f"{candidate} may not run. Tier V does not compare candidates.")
    passed = [r for r in veto if r["candidate"] == candidate and r["kind"] == "comparison"]
    if tier == "V":
        if passed:
            raise CandidateConflict(
                f"{candidate} already used its Tier V pass in regime {regime} "
                f"(decision {passed[-1]['decision']}).")
        return
    if not passed:
        raise CandidateConflict(f"{candidate} has no Tier V comparison in regime {regime}; Tier T is closed.")
    if passed[-1]["decision"] != "Go":
        raise CandidateConflict(
            f"{candidate} was not cleared on Tier V in regime {regime} "
            f"(decision {passed[-1]['decision']}); no Tier T.")
    scored = [r for r in rows if r["tier"] == "T" and r["kind"] == "comparison"]
    if scored:
        raise CandidateConflict(
            f"Tier T in regime {regime} was already scored for {scored[-1]['candidate']}.")
