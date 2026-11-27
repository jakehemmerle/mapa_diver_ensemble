import argparse
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from classical import features, run
from common import confirm_fallback, predictions, ridge, splits
from common.splits import Cell
from ensemble.members import mapa
from synthetic import Provider, make_recording, make_table

SESSION, RATE = "btbank9_0", 1024


@pytest.fixture
def cell():
    """A planted within-session Tier P cell cached under this test's NPEXP_ROOT."""
    table = make_table(RATE, 420, seed=7)
    recordings = {SESSION: make_recording(table, RATE, seed=11)}
    splits.set_provider(Provider({SESSION: table}))
    features.set_recording_source(recordings.__getitem__)
    features.extract(SESSION, {("window", "bipolar"): ("base6", "B1", "B3", "B1~phaserand"),
                               ("session", "bipolar"): ("base6", "B1")}, workers=1)
    return Cell("P", "within", SESSION, SESSION, "speech", 0)


def test_arm_names_and_controls():
    assert run.base_arm().member == "ridge6" and run.base_arm().oof
    assert run.base_arm(mode="session").member == "ridge6@session"
    assert run.base_arm(reference="laplacian").member == "ridge6@laplacian"
    arm = run._block_arm(("B1", "B3"), 0.25)
    assert arm.member == "classical_base6+B1+B3" and arm.weights == (1.0, 0.25, 0.25) and arm.versus == "ridge6"
    assert run._block_arm(("B1",), 0.5, mode="session").versus == "ridge6@session"
    shadows = {a.role: a for a in run.control_arms(arm, 1)}
    assert set(shadows) == {"sham", "rp", "phaserand"}
    assert shadows["sham"].kernels == (("c", "base6", "window", "bipolar"), ("sham", "B1", "window", "bipolar"),
                                       ("sham", "B3", "window", "bipolar"))
    assert shadows["phaserand"].kernels[1:] == (("c", "B1~phaserand", "window", "bipolar"), ("c", "B3", "window", "bipolar"))
    assert all(a.weights == arm.weights and a.kind == "control" and a.versus == "ridge6" for a in shadows.values())
    assert "phaserand" not in {a.role for a in run.control_arms(run._block_arm(("B3",), 1.0), 1)}
    assert arm.hash() != run._block_arm(("B1", "B3"), 0.5).hash()
    assert run.arm_from(arm.config()).kernels == arm.kernels
    stack = run.stacked("mapa_plus_classical_base6+B1", run.MAPA_BASE, ("base6", "B1"), 0.1)
    assert stack.weights == (1.0, 1.0, 0.1, 0.1) and stack.kernels[:2] == (("mapa", "12"), ("mapa", "0"))


def test_run_cell_writes_label_free_scores_for_every_arm(cell, tmp_path):
    predictions = pytest.importorskip("common.predictions")
    candidate = run._block_arm(("B1",), 0.25, role="candidate")
    arms = [run.base_arm(), candidate] + run.control_arms(candidate, 1) + [run._block_arm(("B1",), 0.25, mode="session")]
    results = tmp_path / "results"
    records = run.run_cell((cell, arms, str(results), "base6", False))
    assert [r["status"] for r in records] == ["ok"] * len(arms)
    split = splits.get_cell(cell)
    for arm in arms:
        pred = predictions.read(cell, arm.member, results)
        assert np.array_equal(pred.rows["eval"], split.rows["eval"]) and np.array_equal(pred.rows["select"], split.rows["select"])
        assert ("train" in pred.scores) == arm.oof
        assert pred.meta["weights"] == list(arm.weights) and pred.meta["config_hash"] == arm.hash()
        assert 0 < pred.meta["lam"] and "fit_select_auroc" in pred.meta
        assert pred.meta["select_auroc"] == ridge.auroc(pred.scores["select"], split.labels("select"))
    with np.load(predictions.path(cell, "ridge6", results)) as saved:
        assert set(saved.files) == {"row", "part", "score", "meta"}       # no labels in the file
    base = predictions.read(cell, "ridge6", results)
    assert np.array_equal(base.rows["train"], split.rows["train"])
    assert ridge.auroc(base.scores["train"], split.labels("train")) > 0.8   # out-of-fold scores carry the signal
    assert predictions.read(cell, candidate.member + "~rp", results).meta["rp_exact"] is True
    again = run.run_cell((cell, arms, str(results), "base6", False))
    assert {r["status"] for r in again} == {"existing"}


@pytest.fixture
def confirmation_ridge(monkeypatch):
    # Synthetic only: label access deliberately cannot return eval labels.
    cell = Cell("V", "within", SESSION, SESSION, "speech", 0)
    rng = np.random.default_rng(10)
    train = rng.normal(size=(8, 3))
    train -= train.mean(axis=0)
    select = rng.normal(size=(4, 3))
    evaluate = rng.normal(size=(5, 3))
    kernel = ridge.KernelSet(train @ train.T, {"select": select @ train.T, "eval": evaluate @ train.T})
    labels = {"train": np.tile([0, 1], 4), "select": np.tile([0, 1], 2)}
    rows = {part: np.arange(len(x)) for part, x in zip(splits.PARTS, (train, select, evaluate))}
    blocks = [(rows["train"][4:], rows["train"][:4]), (rows["train"][:4], rows["train"][4:])]
    split = SimpleNamespace(rows=rows, labels=labels.__getitem__, oof_blocks=lambda: blocks)
    monkeypatch.setattr(features, "kernels_and_dim", lambda *_: (kernel, train.shape[1]))
    monkeypatch.setattr(mapa.mapa_features, "kernels", lambda *args, **kwargs: kernel)
    monkeypatch.setattr(mapa.mapa_features, "meta", lambda *_: {})
    with monkeypatch.context() as split_patch:
        split_patch.setattr(splits, "get_cell", lambda _: split)
        yield cell, labels


def _export_ridge(cell, member, root):
    if member == "ridge6":
        run.run_cell((cell, [run.base_arm()], str(root), "base6", False))
    else:
        mapa.export_cell(member, cell, root)
    return predictions.read(cell, member, root)


@pytest.mark.parametrize("member,lam_index", [("ridge6", 13), ("mapa12", 11)])
def test_confirmation_ridge_exports_fixed_predictions(confirmation_ridge, tmp_path, member, lam_index):
    cell, labels = confirmation_ridge
    labels["select"][:] = 0
    pred = _export_ridge(cell, member, tmp_path)
    assert pred.meta["confirm_fallback"] == confirm_fallback.for_select(cell, member, labels["select"])
    assert pred.meta["lam"] == ridge.LAM_MULTS[lam_index]
    assert pred.meta["select_auroc"] is pred.meta["fit_select_auroc"] is None
    for part in splits.PARTS:
        assert np.ptp(pred.scores[part]) > 0


@pytest.mark.parametrize("member", ["ridge6", "mapa12"])
def test_confirmation_ridge_two_class_export_matches_ordinary(confirmation_ridge, tmp_path, member):
    cell, _ = confirmation_ridge
    ordinary = _export_ridge(replace(cell, tier="P"), member, tmp_path)
    scoped = _export_ridge(cell, member, tmp_path)
    assert scoped.meta == dict(ordinary.meta, tier="V")
    assert "confirm_fallback" not in scoped.meta
    for part in splits.PARTS:
        np.testing.assert_array_equal(scoped.rows[part], ordinary.rows[part])
        np.testing.assert_array_equal(scoped.scores[part], ordinary.scores[part])


@pytest.mark.parametrize("one_class_select", [False, True])
def test_confirmation_ridge6_never_exports_training_placeholder(confirmation_ridge, tmp_path, one_class_select):
    cell, labels = confirmation_ridge
    labels["train"][:] = 1
    if one_class_select:
        labels["select"][:] = 0
    with pytest.raises(ridge.Degenerate, match="training labels"):
        _export_ridge(cell, "ridge6", tmp_path)
    assert not predictions.exists(cell, "ridge6", tmp_path)

    ordinary = _export_ridge(replace(cell, tier="P"), "ridge6", tmp_path)
    assert ordinary.meta["status"] == "degenerate"
    assert "confirm_fallback" not in ordinary.meta
    assert set(ordinary.scores) == {"select", "eval"}
    assert all(not np.any(scores) for scores in ordinary.scores.values())


def test_b4_arms_are_not_fit_across_subjects(tmp_path, monkeypatch):
    opts = argparse.Namespace(stages=["extract", "fit", "score"], rp_source="base6", overwrite=False,
                              fit_workers=1, workers=1, min_free_gib=0.0)
    runner = run.Run("c1", "P", opts)
    rows = []
    monkeypatch.setattr(runner, "log", lambda regime, kind, arm, **fields: rows.append((regime, kind, arm, fields)))
    arms = [run._block_arm(("B4",), 1.0), run._block_arm(("B1",), 1.0)]
    assert [a.member for a in runner.available("cross_subject", arms)] == ["classical_base6+B1"]
    assert [a.member for a in runner.available("within", arms)] == [a.member for a in arms]
    runner.available("cross_subject", arms)
    assert len(rows) == 1 and rows[0][3]["decision"] == "not-applicable"   # announced once


def test_weight_choice_is_tier_p_only(tmp_path):
    opts = argparse.Namespace(stages=[], rp_source="base6", overwrite=False, fit_workers=1, workers=1, min_free_gib=0.0)
    for tier in ("V", "T"):
        with pytest.raises(RuntimeError):
            run.Run("c4", tier, opts).choose_weight("within", {1.0: run.base_arm()})
    assert run.WEIGHT_GRID == (0.1, 0.25, 0.5, 1.0)
