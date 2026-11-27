"""The whole ensemble track on the synthetic benchmark: kernels -> member exports -> E0 ->
E1 -> E2 -> E3 -> freeze -> E4 (Tier V veto) -> E5 (Tier T behind the lock)."""
import json

import pytest

pytest.importorskip("sklearn")

from common import ledger, lock, predictions, splits, stats  # noqa: E402
from ensemble import run  # noqa: E402
from ensemble.tests import synthetic  # noqa: E402

REGIME = "cross_subject"


def _run(*argv):
    run.main(list(argv))


def _export(tier, root):
    _run("export", "--tier", tier, "--regime", REGIME, "--members", "mapa12", "mapa0")
    synthetic.write_member("ridge6", tier, REGIME, root, effect=1.1)       # the classical track's export
    _run("e0", "--tier", tier, "--regime", REGIME, "--wave", "1")


def test_end_to_end(root, capsys):
    work = root.parent / "work"
    # Missing kernels are named, and nothing is written.
    with pytest.raises(SystemExit) as stop:
        _run("export", "--tier", "P", "--regime", REGIME, "--members", "mapa12", "--dry-run")
    assert stop.value.code == 2 and "MISSING INPUT" in capsys.readouterr().out
    synthetic.write_kernels("P", REGIME, work)
    synthetic.write_kernels("T", REGIME, work)

    _export("P", root)
    state = json.loads((root / "ensemble" / f"e0_P_{REGIME}.json").read_text())
    assert all(state[m]["passed"] and state[m]["max_abs_gap"] <= 1e-6 for m in ("mapa12", "mapa0", "ridge6"))

    _run("e1", "--regime", REGIME, "--wave", "1", "--dry-run")
    _run("e1", "--regime", REGIME, "--wave", "1")
    screen = json.loads((root / "ensemble" / f"e1_{REGIME}_w1.json").read_text())
    assert len(screen["subsets"]) == 4 and screen["passed"]
    assert "ridge6" in screen["best"]["members"]                    # the member with independent errors
    assert screen["self_ensemble"]["ran"] and screen["self_ensemble"]["gain"] < 0.02
    assert set(screen["mapa_pretraining_split"]) == {"held_out_5_0", "pretraining_sessions"}

    _run("e2", "--regime", REGIME, "--wave", "1")
    ladder = json.loads((root / "ensemble" / f"e2_{REGIME}_w1.json").read_text())
    # (b) is adopted only with a gain of 0.003 and a passing shuffled-label control
    assert (ladder["rung"] == "b") == (ladder["stacker_gain"] >= 0.003 and ladder["shuffled_passed"])

    _run("e3", "--regime", REGIME, "--wave", "1")
    design = json.loads((root / "ensemble" / f"e3_{REGIME}_w1.json").read_text())
    assert design["decision"] == "Go" and design["controls_pass"] and design["self_ensemble_pass"]
    assert design["gain_over_best_member"] - design["self_ensemble"]["gain"] >= 0.003

    # Tier V and Tier T are closed until a candidate is frozen.
    for phase in ("e4", "e5"):
        with pytest.raises(SystemExit) as stop:
            _run(phase, "--regime", REGIME, "--dry-run")
        assert stop.value.code == 2
    _run("freeze", "--regime", REGIME)
    frozen = lock.frozen(REGIME, root)
    candidate = lock.candidate_id(frozen)
    assert frozen["track"] == "ensemble" and frozen["member"] == design["candidate"]

    with pytest.raises(SystemExit) as stop:                          # Tier V exports are missing
        _run("e4", "--regime", REGIME)
    assert stop.value.code == 2
    _export("V", root)
    _run("e4", "--regime", REGIME)
    veto = json.loads((root / "ensemble" / f"e4_{REGIME}.json").read_text())
    assert veto["decision"] == "Go" and veto["n_subjects"] == 5
    with pytest.raises(SystemExit) as stop:                          # one pass only
        _run("e4", "--regime", REGIME)
    assert stop.value.code == 3 and "already used its Tier V pass" in capsys.readouterr().err

    # Tier T: exports need no eval label; scoring reads them once, logged.
    _export("T", root)
    assert lock.access_log(root) == []
    _run("e5", "--regime", REGIME)
    score = json.loads((root / "ensemble" / f"e5_{REGIME}.json").read_text())
    assert score["candidate"] == candidate and score["delta"] > 0.01
    log = lock.access_log(root)
    assert len(log) == 1 and log[0]["purpose"] == "score" and log[0]["candidate"] == candidate
    lock._granted.clear()
    stats._label_cache.clear()
    with pytest.raises(SystemExit) as stop:                          # Tier T is scored once
        _run("e5", "--regime", REGIME)
    assert stop.value.code == 3

    # A reproduction run is a separate, marked access.
    _run("reproduce", "--regime", REGIME, "--member", "mapa12")
    log = lock.access_log(root)
    assert [entry["purpose"] for entry in log] == ["score", "reproduction"] and log[1]["member"] == "mapa12"
    report = json.loads((root / "reproduction" / f"mapa12_{REGIME}.json").read_text())
    assert report["cells"] == 150 and report["published"] == 0.6083 and 0.5 < report["macro"] < 1.0

    ledger.verify(root)
    rows = ledger.read(root)
    assert {r["phase"] for r in rows} >= {"E0", "E1", "E2", "E3", "FREEZE", "E4", "E5", "REPRO"}
    veto_rows = [r for r in rows if r["tier"] == "V" and r["kind"] == "comparison"]
    assert len(veto_rows) == 1 and veto_rows[0]["candidate"] == candidate and veto_rows[0]["decision"] == "Go"
    final = [r for r in rows if r["tier"] == "T" and r["kind"] == "comparison"]
    assert len(final) == 1 and final[0]["decision"] == "Reported" and final[0]["commit"] == "0123456789abcdef"
    # no prediction file carries labels
    cell = splits.list_cells("T", REGIME)[0]
    assert set(predictions.read(cell, frozen["member"], root).scores) >= {"select", "eval"}


def test_uniform_confirmation_requires_independent_parity_and_fixed_design(root):
    spec = {"members": ["diver", "mapa12"], "normalization": "select_cdf", "rung": "a", "penalty": None}
    design = {"track": "ensemble", "regime": "within", "spec": spec}
    assert not run.quick_confirmation(design, root)
    path = root / "parity" / "diver_uniform_confirmation.json"
    path.parent.mkdir(parents=True)
    receipt = {"passed": True, "tier": "P", "independent_full_heads": True, "tolerance": 1e-6,
               "regimes": {"within": {"passed": True}}}
    path.write_text(json.dumps(receipt))
    assert run.quick_confirmation(design, root)
    assert not run.quick_confirmation(dict(design, regime="cross_session"), root)
    assert not run.quick_confirmation(dict(design, spec=dict(spec, rung="b")), root)
    assert not run.quick_confirmation(dict(design, track="classical"), root)
    path.write_text(json.dumps(dict(receipt, passed=False)))
    assert not run.quick_confirmation(design, root)


def test_a_second_candidate_cannot_use_tier_v(root, capsys):
    work = root.parent / "work"
    synthetic.write_kernels("P", REGIME, work)
    synthetic.write_kernels("T", REGIME, work)
    _export("P", root)
    for phase in ("e1", "e2", "e3"):
        _run(phase, "--regime", REGIME, "--wave", "1")
    _run("freeze", "--regime", REGIME)
    _export("V", root)
    ledger.append({"tier": "V", "regime": REGIME, "phase": "C4", "kind": "comparison", "arm": "classical",
                   "candidate": "classical:ffff", "decision": "Veto"}, root)
    with pytest.raises(SystemExit) as stop:
        _run("e4", "--regime", REGIME)
    assert stop.value.code == 3 and "classical:ffff" in capsys.readouterr().err


def test_e1_failure_stops_the_ladder(root, capsys):
    # Three members that share one noise source: fusion cannot beat the best of them.
    for member in ("mapa12", "mapa0", "ridge6"):
        synthetic.write_member(member, "P", REGIME, root, effect=1.0, noise_of="shared")
    _run("e0", "--tier", "P", "--regime", REGIME, "--wave", "1")
    _run("e1", "--regime", REGIME, "--wave", "1")
    screen = json.loads((root / "ensemble" / f"e1_{REGIME}_w1.json").read_text())
    assert not screen["passed"] and screen["best"]["decision"] == "Stop"
    assert not screen["self_ensemble"]["ran"]                       # no kernels: recorded, not hidden
    with pytest.raises(SystemExit) as stop:
        _run("e2", "--regime", REGIME, "--wave", "1")
    assert stop.value.code == 3 and "E1 failed" in capsys.readouterr().err
    rows = ledger.read(root)
    assert any(r["phase"] == "E2" and r["decision"] == "not run" for r in rows)
    assert any(r["arm"] == "self_ensemble" and r["decision"] == "not run" for r in rows)
