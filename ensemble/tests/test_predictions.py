import numpy as np
import pytest

from common import predictions, splits, stats
from ensemble import fuse
from ensemble.tests import synthetic


def _cell():
    return splits.list_cells("P", "cross_subject")[0]


def test_round_trip(root):
    cell = _cell()
    split = splits.get_cell(cell)
    rng = np.random.default_rng(0)
    scores = {part: rng.normal(size=len(split.rows[part])).astype(np.float32) for part in predictions.PARTS}
    file = predictions.write(cell, "m", split.rows, scores, {"lam": 1.0, "select_auroc": 0.5}, root)
    assert file == root / "preds" / "P" / "cross_subject" / "m" / f"{cell.key()}.npz"
    with np.load(file) as raw:
        assert set(raw.files) == {"row", "part", "score", "meta"}
        assert raw["row"].dtype == np.int32 and raw["part"].dtype == np.uint8 and raw["score"].dtype == np.float32
    back = predictions.read(cell, "m", root)
    for part in predictions.PARTS:
        assert np.array_equal(back.rows[part], split.rows[part])
        assert np.array_equal(back.scores[part].astype(np.float32), scores[part])
    assert back.meta["lam"] == 1.0 and back.meta["member"] == "m" and back.meta["task"] == cell.task


def test_labels_are_never_stored(root):
    cell = _cell()
    split = splits.get_cell(cell)
    scores = {"select": np.zeros(len(split.rows["select"]))}
    with pytest.raises(ValueError, match="labels"):
        predictions.write(cell, "m", split.rows, scores, {"labels": [0, 1]}, root)


def test_writer_rejects_bad_input(root):
    cell = _cell()
    split = splits.get_cell(cell)
    n = len(split.rows["select"])
    with pytest.raises(ValueError, match="non-finite"):
        predictions.write(cell, "m", split.rows, {"select": np.full(n, np.nan)}, {}, root)
    with pytest.raises(ValueError, match="duplicate"):
        predictions.write(cell, "m", {"select": np.zeros(n, dtype=np.int64)}, {"select": np.zeros(n)}, {}, root)
    with pytest.raises(ValueError, match="differ"):
        predictions.write(cell, "m", split.rows, {"select": np.zeros(n - 1)}, {}, root)


def test_align_reorders_and_detects_mismatch():
    rows = np.array([5, 3, 9])
    assert np.array_equal(predictions.align(np.array([0.5, 0.3, 0.9]), rows, np.array([3, 5, 9]), "x"), [0.3, 0.5, 0.9])
    with pytest.raises(predictions.RowSetMismatch):
        predictions.align(np.zeros(3), rows, np.array([3, 5, 8]), "x")
    with pytest.raises(predictions.RowSetMismatch):
        predictions.align(np.zeros(3), rows, np.array([3, 5]), "x")


def _drop_one_eval_row(member, root):
    cell = _cell()
    pred = predictions.read(cell, member, root)
    rows = {part: pred.rows[part][:-1] if part == "eval" else pred.rows[part] for part in pred.rows}
    scores = {part: pred.scores[part][:-1] if part == "eval" else pred.scores[part] for part in pred.scores}
    predictions.write(cell, member, rows, scores, {"select_auroc": 0.5}, root)


def test_compare_fails_loudly_on_row_set_mismatch(root):
    for member in ("a", "b"):
        synthetic.write_member(member, "P", "cross_subject", root)
    stats.compare("a", "b", "P", "cross_subject", root)
    _drop_one_eval_row("b", root)
    with pytest.raises(predictions.RowSetMismatch):
        stats.compare("a", "b", "P", "cross_subject", root)


def test_fusion_fails_loudly_on_row_set_mismatch(root):
    for member in ("a", "b"):
        synthetic.write_member(member, "P", "cross_subject", root)
    _drop_one_eval_row("b", root)
    with pytest.raises(predictions.RowSetMismatch):
        fuse.fuse(["a", "b"], "a", "P", "cross_subject", root)
