import json

import pytest

from common import ledger, lock, splits

ROW = {"tier": "P", "regime": "within", "phase": "E1", "kind": "comparison", "arm": "ens", "decision": "Go"}


def test_ledger_is_append_only(root):
    ledger.append(dict(ROW, delta=0.011), root)
    first = (root / "ledger.csv").read_bytes()
    ledger.append(dict(ROW, arm="other", decision="Stop", extra={"why": "failed arms stay"}), root)
    ledger.append(dict(ROW, arm="third", unknown_key=3), root)
    data = (root / "ledger.csv").read_bytes()
    assert data.startswith(first) and len(data) > len(first)        # earlier bytes are untouched
    rows = ledger.read(root)
    assert [r["arm"] for r in rows] == ["ens", "other", "third"]
    assert rows[0]["commit"] == "0123456789abcdef" and rows[0]["delta"] == "0.011"
    assert json.loads(rows[2]["extra"]) == {"unknown_key": 3}
    ledger.verify(root)
    assert not [name for name in dir(ledger) if any(word in name.lower() for word in ("delete", "update", "rewrite", "remove"))]


def test_ledger_detects_an_edited_or_removed_row(root):
    for arm in ("a", "b", "c"):
        ledger.append(dict(ROW, arm=arm), root)
    file = root / "ledger.csv"
    lines = file.read_bytes().splitlines(keepends=True)
    file.write_bytes(b"".join(lines[:2] + lines[3:]))               # drop the second row
    with pytest.raises(ledger.LedgerError, match="does not chain"):
        ledger.verify(root)
    file.write_bytes(b"".join(lines).replace(b",Go,", b",Stop,", 1))
    with pytest.raises(ledger.LedgerError, match="does not chain"):
        ledger.verify(root)


def test_ledger_rejects_incomplete_rows(root):
    with pytest.raises(ledger.LedgerError, match="lacks"):
        ledger.append({"tier": "P", "regime": "within"}, root)
    with pytest.raises(ledger.LedgerError, match="kind"):
        ledger.append(dict(ROW, kind="whatever"), root)
    with pytest.raises(ledger.LedgerError, match="assigned by the ledger"):
        ledger.append(dict(ROW, prev="0"), root)
    assert not (root / "ledger.csv").exists()


def _lock(root, grants, commit="0123456789abcdef"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "LOCK.json").write_text(json.dumps({"commit": commit, "grants": grants}))


def _eval_labels():
    return splits.get_cell(splits.list_cells("T", "within")[0]).labels("eval")


def test_tier_t_score_needs_a_frozen_candidate(root):
    lock.require_test_unlocked("P", "within", root)                  # tiers P and V are never gated
    lock.require_test_unlocked("V", "within", root)
    with pytest.raises(lock.TestLocked, match="no frozen candidate"):
        lock.require_test_unlocked("T", "within", root)               # default purpose is "score"
    with pytest.raises(lock.TestLocked):
        _eval_labels()                                               # the only route to Tier T eval labels
    cell = splits.get_cell(splits.list_cells("T", "within")[0])
    assert len(cell.labels("train")) and len(cell.labels("select")) and len(cell.rows["eval"])
    assert lock.access_log(root) == []


def test_reproduction_access_is_logged_once_per_process(root, monkeypatch):
    monkeypatch.setenv("NPEXP_TEST_PURPOSE", "reproduction")
    monkeypatch.setenv("NPEXP_TEST_MEMBER", "ridge6")
    assert len(_eval_labels()) > 0                                   # the bare three-argument call in common.splits
    lock.require_test_unlocked("T", "within", root, purpose="reproduction", member="ridge6")
    log = lock.access_log(root)
    assert len(log) == 1
    assert log[0]["purpose"] == "reproduction" and log[0]["member"] == "ridge6" and log[0]["regime"] == "within"
    assert log[0]["commit"] == "0123456789abcdef"
    lock._granted.clear()                                            # a second process logs again
    lock.require_test_unlocked("T", "within", root, purpose="reproduction", member="ridge6")
    assert len(lock.access_log(root)) == 2


def _propose(root, regime, track, gain, decision, controls=True):
    return lock.propose(regime, track, f"{track}_member", {"design": track}, gain, decision, controls, root)


def test_one_candidate_is_frozen_per_regime(root):
    with pytest.raises(lock.FreezeError, match="no frozen candidate"):
        lock.frozen("within", root)
    _propose(root, "within", "classical", 0.02, "Hold", controls=False)       # larger gain, not eligible
    _propose(root, "within", "ensemble", 0.012, "Go")
    _propose(root, "cross_session", "ensemble", None, "Inherited")
    with pytest.raises(lock.FreezeError, match="no frozen candidate"):
        lock.freeze("cross_session", root)                                    # within is not frozen yet
    design = lock.freeze("within", root)
    assert design["track"] == "ensemble" and len(design["considered"]) == 2
    assert lock.frozen("within", root)["config_hash"] == design["config_hash"]
    assert lock.candidate_id(design) == f"ensemble:{design['config_hash']}"
    with pytest.raises(lock.FreezeError, match="frozen once"):
        lock.freeze("within", root)
    with pytest.raises(lock.FreezeError, match="already frozen"):
        _propose(root, "within", "classical", 0.5, "Go")
    assert lock.freeze("cross_session", root)["track"] == "ensemble"          # inherits the within track
    _propose(root, "cross_subject", "ensemble", 0.001, "Stop")
    with pytest.raises(lock.FreezeError, match="no proposal"):
        lock.freeze("cross_subject", root)
    file = root / "frozen" / "within.json"
    file.write_text(file.read_text().replace('"design": "ensemble"', '"design": "edited"'))
    with pytest.raises(lock.FreezeError, match="edited after freezing"):
        lock.frozen("within", root)


def test_tier_v_and_t_admit_one_candidate_once(root):
    mine, other = "ensemble:aaaa", "classical:bbbb"
    veto = {"tier": "V", "regime": "within", "phase": "E4", "kind": "comparison"}
    ledger.guard_candidate(root, "V", "within", mine)
    with pytest.raises(ledger.CandidateConflict, match="no Tier V comparison"):
        ledger.guard_candidate(root, "T", "within", mine)
    ledger.append(dict(veto, kind="member", arm="mapa12", candidate=other), root)
    with pytest.raises(ledger.CandidateConflict, match="different|already holds"):
        ledger.guard_candidate(root, "V", "within", mine)
    ledger.guard_candidate(root, "V", "cross_subject", mine)                  # other regimes are independent
    ledger.append(dict(veto, regime="cross_subject", arm="ens", candidate=mine, decision="Veto"), root)
    with pytest.raises(ledger.CandidateConflict, match="already used its Tier V pass"):
        ledger.guard_candidate(root, "V", "cross_subject", mine)
    with pytest.raises(ledger.CandidateConflict, match="not cleared"):
        ledger.guard_candidate(root, "T", "cross_subject", mine)
    ledger.append(dict(veto, regime="cross_session", arm="ens", candidate=mine, decision="Go"), root)
    ledger.guard_candidate(root, "T", "cross_session", mine)
    ledger.append(dict(veto, tier="T", phase="E5", regime="cross_session", arm="ens", candidate=mine,
                       decision="Reported"), root)
    with pytest.raises(ledger.CandidateConflict, match="already scored"):
        ledger.guard_candidate(root, "T", "cross_session", mine)


def test_score_access_needs_the_frozen_candidate_and_a_tier_v_go(root):
    _propose(root, "within", "ensemble", 0.012, "Go")
    design = lock.freeze("within", root)
    candidate = lock.candidate_id(design)
    with pytest.raises(lock.TestLocked, match="frozen candidate"):
        lock.require_test_unlocked("T", "within", root, config_hash="not-the-frozen-one")
    with pytest.raises(lock.TestLocked, match="no Tier V comparison"):
        lock.require_test_unlocked("T", "within", root)
    ledger.append({"tier": "V", "regime": "within", "phase": "E4", "kind": "comparison", "arm": "ens",
                   "candidate": candidate, "decision": "Go"}, root)
    assert lock.access_log(root) == []
    assert len(_eval_labels()) > 0
    log = lock.access_log(root)
    assert len(log) == 1 and log[0]["purpose"] == "score" and log[0]["candidate"] == candidate
