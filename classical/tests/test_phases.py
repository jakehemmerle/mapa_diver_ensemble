"""End-to-end run of the Tier P decision phases on a small synthetic universe.

Three permitted sessions (two of one subject), two tasks, planted phase-locked responses.
The numbers are meaningless; the test pins the plumbing: every arm scores every cell, the
weight is chosen and recorded, controls and decisions reach the ledger, cross-session
copies the within-session choice, and C2 writes a proposal.
"""
import json

import numpy as np
import pytest

import common
from classical import features, run
from common import ridge, splits
from synthetic import Provider, make_recording, make_table

SESSIONS = ("btbank2_5", "btbank2_6", "btbank9_0")


@pytest.fixture
def universe(monkeypatch):
    stats = pytest.importorskip("common.stats")
    monkeypatch.setattr(splits, "DEV_SESSIONS", SESSIONS)
    monkeypatch.setattr(splits, "TASKS", ("speech", "pitch"))
    monkeypatch.setitem(stats.THRESHOLDS, ("P", "within"), {"go": 0.010, "up": 2, "of": 2, "hold": 0.005})
    monkeypatch.setitem(stats.THRESHOLDS, ("P", "cross_subject"), {"go": 0.015, "up": 1, "of": 1, "hold": 0.008})
    tables = {s: make_table(splits.native_rate(s), 220, seed=i) for i, s in enumerate(SESSIONS)}
    recordings = {s: make_recording(tables[s], splits.native_rate(s), seed=10 + i) for i, s in enumerate(SESSIONS)}
    splits.set_provider(Provider(tables))
    features.set_recording_source(recordings.__getitem__)
    return ["--workers", "1", "--fit-workers", "1", "--min-free-gib", "0"]


def _fake_mapa(regimes):
    """Stand-ins for the ensemble's outputs: MAPA kernels (taps 0 and 12) and the mapa12 member."""
    from common import mapa_features, predictions
    for regime in regimes:
        for cell in splits.list_cells("P", regime):
            split = splits.get_cell(cell)
            target_rows = np.concatenate([split.rows["select"], split.rows["eval"]])
            for tap, block in ((0, "B3"), (12, "env5")):
                m = features.cell_matrices(cell, block, "window", "bipolar")
                target = np.concatenate([m["select"], m["eval"]])
                mapa_features.save(mapa_features.kernel_file("P", regime, cell.key(), tap), train=m["train"] @ m["train"].T,
                                   cross=target @ m["train"].T, train_rows=split.rows["train"], target_rows=target_rows,
                                   meta={"synthetic": True}, target=target @ target.T if tap == 0 else None)
            fit = ridge.fit_predict([mapa_features.kernels(cell, 12)], split.labels("train"), split.labels("select"))
            predictions.write(cell, "mapa12", split.rows, {p: fit["scores"][p] for p in ("select", "eval")},
                              {"synthetic": True}, common.results_root())


def test_tier_p_phases_end_to_end(universe):
    from common import ledger, lock, predictions
    root = common.results_root()
    assert run.main(["p0", "--regimes", "within"] + universe) == 0
    assert run.main(["c1", "--rp-source", "base6"] + universe) == 0
    ledger.verify(root)
    rows = ledger.read(root)
    assert {r["kind"] for r in rows} >= {"info", "member", "control"}
    assert all(r["tier"] == "P" for r in rows)

    within = json.loads((root / "classical" / "c1_within.json").read_text())
    assert set(within["block_weights"]) == set(run.ADDED_BLOCKS)
    assert set(within["block_weights"].values()) <= set(run.WEIGHT_GRID)
    assert set(within["reference_factor"]) == {"ridge6", "ridge6@laplacian", "ridge6@shaft_mean", "ridge6@session"}
    chosen = [r for r in rows if r["decision"] == "weight-chosen" and r["regime"] == "within"]
    assert len(chosen) >= len(run.ADDED_BLOCKS)
    for block, weight in within["block_weights"].items():
        cell = splits.list_cells("P", "within")[0]
        assert predictions.read(cell, f"classical_base6+{block}", root).meta["weights"] == [1.0, weight]
        for control in ("~sham", "~rp"):
            assert predictions.exists(cell, f"classical_base6+{block}{control}", root)
    assert within["block_controls"]["B1"]["needs_phase"] in (True, False)

    cross_subject = json.loads((root / "classical" / "c1_cross_subject.json").read_text())
    assert "B4" not in cross_subject["block_weights"]
    assert any(r["decision"] == "not-applicable" and r["regime"] == "cross_subject" for r in rows)
    cross_session = json.loads((root / "classical" / "c1_cross_session.json").read_text())
    assert cross_session["copied_from_within"] and cross_session["kept_blocks"] == within["kept_blocks"]
    assert cross_session["block_weights"] == within["block_weights"]
    kept = json.loads((root / "classical" / "kept_within.json").read_text())
    assert kept["member"] == "mapa_plus_classical_" + "+".join(["base6"] + kept["kept_blocks"])

    # C2 stops, naming the input, while the ensemble's MAPA kernels are absent.
    assert run.main(["c2", "--regimes", "within"] + universe) == 2
    assert run.main(["c2", "--regimes", "within", "--dry-run"] + universe) == 2

    _fake_mapa(splits.REGIMES)
    assert run.main(["c2"] + universe) == 0
    assert run.main(["c3"] + universe) == 0
    ledger.verify(root)
    for regime in splits.REGIMES:
        saved = json.loads((root / "classical" / f"c2_{regime}.json").read_text())
        assert saved["weight"] in run.WEIGHT_GRID and saved["baseline"] in run.BASELINES
        proposal = [p for p in lock.proposals(regime, root) if p["track"] == "classical"][0]
        assert proposal["member"] == saved["member"] and proposal["spec"]["weight"] == saved["weight"]
        assert proposal["spec"]["weights"][:2] == [1.0, 1.0] and proposal["spec"]["window_mode"] == "window"
        assert isinstance(proposal["controls_pass"], bool)
        cell = splits.list_cells("P", regime)[0]
        stack = predictions.read(cell, saved["member"], root)
        assert "train" in stack.scores                                   # out-of-fold scores for the ensemble's stacker
        assert predictions.read(cell, saved["member"] + "~rp", root).meta["rp_exact"] is True
        assert (root / "classical" / f"controls_{regime}.json").exists()
    c2 = json.loads((root / "classical" / "c2_cross_session.json").read_text())
    assert c2["weight"] == json.loads((root / "classical" / "c2_within.json").read_text())["weight"]
    comparisons = [r for r in ledger.read(root) if r["kind"] == "comparison"]
    assert {r["regime"] for r in comparisons} == set(splits.REGIMES)
    assert all(not r["candidate"] for r in ledger.read(root))            # Tier P rows never claim a Tier V/T slot


@pytest.mark.parametrize("decision,controls,expected", [
    ("Stop", True, ("base6",)), ("Hold", False, ("base6",)),
    ("Hold", True, ("base6", "B1")), ("Go", True, ("base6", "B1")),
])
def test_c2_uses_completed_kept_set_gate(tmp_path, decision, controls, expected):
    runner = run.Run("c2", "P", None)
    runner.out = tmp_path
    runner.save("c1_within.json", {"decision": decision, "controls": {"controls_pass": controls},
                                   "kept_blocks": ["B1"]})
    for regime in ("within", "cross_session"):
        assert run.c2_design(runner, regime)[0] == expected
    with pytest.raises(run.Missing):
        run.c2_design(runner, "cross_subject")
    runner.save("c1_within.json", {"decision": "Hold", "kept_blocks": ["B1"]})
    with pytest.raises(run.Missing):
        run.c2_design(runner, "within")
