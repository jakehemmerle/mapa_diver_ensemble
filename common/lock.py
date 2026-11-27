"""Tier T access log and the one frozen candidate per regime.

Test labels of the 12 evaluation sessions are reachable only through
:func:`require_test_unlocked`. There is no lock file: every Tier T label read appends
one JSON line to ``<root>/test_access.log`` (one line per process, regime, purpose and
member). ``purpose`` is ``score`` (the scoring of the regime's frozen candidate) or
``reproduction`` (parity of a published entry or member). A ``score`` read is refused
for any candidate other than the one already scored in that regime: evaluation-session
test scores never choose between designs.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import ledger

PURPOSES = ("score", "reproduction")
TRACKS = ("classical", "ensemble")
_granted: set[tuple] = set()


class TestLocked(PermissionError):
    """Tier T labels were requested for a second candidate in a regime."""

    __test__ = False  # not a pytest class


class FreezeError(RuntimeError):
    pass


def _refuse(reason: str):
    raise TestLocked(f"Tier T test labels are locked: {reason}")


def require_test_unlocked(tier: str, regime: str, root: Path, *, purpose: str | None = None,
                          member: str | None = None, config_hash: str | None = None) -> None:
    """Return silently for tiers P and V. For tier T log the access; a ``score`` read
    must be for the regime's frozen candidate and no other candidate may have been scored."""
    if tier != "T":
        return
    root = Path(root)
    purpose = purpose or os.environ.get("NPEXP_TEST_PURPOSE") or "score"
    member = member or os.environ.get("NPEXP_TEST_MEMBER") or None
    if purpose not in PURPOSES:
        _refuse(f"unknown purpose {purpose!r}")
    candidate = None
    if purpose == "score":
        try:
            design = frozen(regime, root)
        except FreezeError as error:
            _refuse(str(error))
        if config_hash is not None and config_hash != design["config_hash"]:
            _refuse(f"the caller runs config {config_hash}, the frozen candidate is {design['config_hash']}")
        candidate = candidate_id(design)
        config_hash = design["config_hash"]
    key = (str(root.resolve()), regime, purpose, member, config_hash)
    if key in _granted:
        return
    if purpose == "score":
        try:
            ledger.guard_candidate(root, "T", regime, candidate)
        except ledger.CandidateConflict as error:
            _refuse(str(error))
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tier": "T",
             "regime": regime, "purpose": purpose, "member": member,
             "commit": ledger.commit(), "config_hash": config_hash, "candidate": candidate,
             "pid": os.getpid(), "argv": sys.argv}
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / "test_access.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    _granted.add(key)


def access_log(root: Path) -> list[dict]:
    file = Path(root) / "test_access.log"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.is_file() else []


# --- one frozen candidate per regime -------------------------------------------------

def candidate_id(design: dict) -> str:
    return f"{design['track']}:{design['config_hash']}"


def _frozen_dir(root: Path) -> Path:
    return Path(root) / "frozen"


def propose(regime: str, track: str, member: str, spec: dict, tier_p_gain: float | None,
            decision: str, controls_pass: bool, root: Path) -> Path:
    """Record a track's Tier P design for a regime. Replaceable until the regime is frozen."""
    if track not in TRACKS:
        raise FreezeError(f"track must be one of {TRACKS}")
    if (_frozen_dir(root) / f"{regime}.json").exists():
        raise FreezeError(f"regime {regime} is already frozen; proposals are closed")
    body = {"regime": regime, "track": track, "member": member, "spec": spec,
            "tier_p_gain": tier_p_gain, "decision": decision, "controls_pass": bool(controls_pass),
            "commit": ledger.commit(),
            "config_hash": ledger.config_hash({"track": track, "member": member, "spec": spec})}
    out = _frozen_dir(root) / "proposals" / f"{regime}.{track}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    os.replace(temp, out)
    return out


def proposals(regime: str, root: Path) -> list[dict]:
    folder = _frozen_dir(root) / "proposals"
    return [json.loads(p.read_text()) for p in sorted(folder.glob(f"{regime}.*.json"))]


def eligible(proposal: dict) -> bool:
    return proposal["decision"] == "Go" or (proposal["decision"] == "Hold" and proposal["controls_pass"])


def freeze(regime: str, root: Path) -> dict:
    """Write ``frozen/<regime>.json`` once: the eligible proposal with the largest Tier P gain.

    Cross-session has no usable Tier P signal: it takes the track frozen for within-session
    and that track's cross-session proposal (the same design, restated for the regime).
    """
    found = proposals(regime, root)
    if regime == "cross_session":
        track = frozen("within", root)["track"]
        chosen = [p for p in found if p["track"] == track]
        if not chosen:
            raise FreezeError(f"within-session froze track {track}, which has no cross_session proposal")
        choice = chosen[0]
    else:
        able = [p for p in found if eligible(p)]
        if not able:
            raise FreezeError(f"no proposal for {regime} is at Go, or at Hold with clean controls: "
                              f"{[(p['track'], p['decision'], p['controls_pass']) for p in found]}")
        choice = max(able, key=lambda p: p["tier_p_gain"])
    out = _frozen_dir(root) / f"{regime}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = dict(choice, frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                frozen_by_commit=ledger.commit(),
                considered=[{k: p[k] for k in ("track", "member", "tier_p_gain", "decision",
                                               "controls_pass", "config_hash")} for p in found])
    try:
        with out.open("x") as handle:
            handle.write(json.dumps(body, indent=1, sort_keys=True) + "\n")
    except FileExistsError:
        raise FreezeError(f"{out} exists; a regime is frozen once") from None
    return body


def frozen(regime: str, root: Path) -> dict:
    file = _frozen_dir(root) / f"{regime}.json"
    if not file.is_file():
        raise FreezeError(f"no frozen candidate for regime {regime} ({file} is absent)")
    design = json.loads(file.read_text())
    expected = ledger.config_hash({k: design[k] for k in ("track", "member", "spec")})
    if design["config_hash"] != expected:
        raise FreezeError(f"{file} was edited after freezing (config hash does not match its content)")
    return design
