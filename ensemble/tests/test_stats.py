import itertools

import numpy as np
import pytest

from common import stats
from ensemble.tests import synthetic


def test_auroc_matches_pair_counting():
    rng = np.random.default_rng(1)
    scores = np.round(rng.normal(size=60), 1)          # rounding creates ties
    labels = rng.integers(0, 2, 60)
    pos, neg = scores[labels == 1], scores[labels == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert stats.auroc(scores, labels) == pytest.approx(brute, abs=1e-12)
    assert np.isnan(stats.auroc(scores, np.ones(60)))


def test_auroc_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    scores, labels = rng.normal(size=500).astype(np.float32), rng.integers(0, 2, 500)
    assert stats.auroc(scores, labels) == pytest.approx(metrics.roc_auc_score(labels, scores), abs=1e-12)


def _brute_sign_flip(deltas):
    deltas = np.asarray(deltas, dtype=float)
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(deltas)):
        total += 1
        hits += np.mean(np.array(signs) * deltas) >= deltas.mean() - 1e-12
    return hits / total


@pytest.mark.parametrize("n", [1, 2, 5, 6, 7, 8])
def test_sign_flip_is_exact(n):
    rng = np.random.default_rng(n)
    deltas = rng.normal(0.01, 0.02, size=n)
    assert stats.sign_flip_p(deltas) == _brute_sign_flip(deltas)


def test_sign_flip_known_values():
    assert stats.sign_flip_p(np.full(8, 0.01)) == 1 / 256          # only the identity reaches the mean
    assert stats.sign_flip_p(np.array([0.02, 0.01, 0.03, 0.01, 0.02, 0.04, 0.01])) == 1 / 128
    assert stats.sign_flip_p(np.array([0.01, -0.01])) == 0.75
    assert stats.sign_flip_p(-np.full(6, 0.01)) == 1.0


def test_holm():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


@pytest.mark.parametrize("tier,regime,delta,up,n,expected", [
    ("P", "within", 0.010, 7, 8, "Go"),
    ("P", "within", 0.0101, 8, 8, "Go"),
    ("P", "within", 0.020, 6, 8, "Hold"),         # above Go but too few subjects up
    ("P", "within", 0.0099, 8, 8, "Hold"),
    ("P", "within", 0.005, 4, 8, "Hold"),
    ("P", "within", 0.0049, 8, 8, "Stop"),
    ("P", "within", -0.01, 0, 8, "Stop"),
    ("P", "cross_subject", 0.015, 6, 7, "Go"),
    ("P", "cross_subject", 0.015, 5, 7, "Hold"),
    ("P", "cross_subject", 0.008, 7, 7, "Hold"),
    ("P", "cross_subject", 0.0079, 7, 7, "Stop"),
    ("P", "cross_session", 0.05, 2, 2, "Descriptive"),
    ("V", "within", 0.005, 5, 6, "Go"),
    ("V", "within", 0.005, 4, 6, "Veto"),
    ("V", "within", 0.0049, 6, 6, "Veto"),
    ("V", "cross_session", 0.006, 6, 6, "Go"),
    ("V", "cross_subject", 0.010, 4, 5, "Go"),
    ("V", "cross_subject", 0.0099, 5, 5, "Veto"),
    ("V", "cross_subject", 0.02, 3, 5, "Veto"),
    ("T", "within", 0.5, 6, 6, "Reported"),
])
def test_threshold_table(tier, regime, delta, up, n, expected):
    assert stats.decide(tier, regime, delta, up, n) == expected


def test_decide_refuses_wrong_subject_count():
    with pytest.raises(ValueError, match="expects 8 subjects"):
        stats.decide("P", "within", 0.02, 7, 7)


def test_compare_hierarchy_and_decision(root):
    synthetic.write_member("strong", "P", "within", root, effect=1.5)
    synthetic.write_member("weak", "P", "within", root, effect=0.5)
    result = stats.compare("strong", "weak", "P", "within", root)
    assert result.n_subjects == 8 and result.subjects_up == 8 and result.decision == "Go"
    assert result.p_signflip == 1 / 256
    assert result.ci_lo > 0 and result.ci_lo < result.delta < result.ci_hi
    assert result.delta == pytest.approx(np.mean(list(result.per_subject.values())))
    assert result.score - result.baseline_score == pytest.approx(result.delta)
    assert len(result.per_task) == 15 and all(v["p_holm"] >= v["p"] for v in result.per_task.values())
    assert set(result.leave_one_subject_out) == set(result.per_subject)
    # subject 2 has five Tier P sessions: its delta is the mean over tasks of session means
    two = [d for s, d in result.per_session.items() if s.startswith("btbank2_")]
    assert result.per_subject["2"] == pytest.approx(np.mean(two))
    reverse = stats.compare("weak", "strong", "P", "within", root)
    assert reverse.decision == "Stop" and reverse.p_signflip == 1.0 and reverse.delta == pytest.approx(-result.delta)
    row = result.row("E1")
    assert row["tier"] == "P" and row["arm"] == "strong" and row["decision"] == "Go"


def test_sub_analyses_get_no_decision(root):
    for member, effect in (("strong", 1.5), ("weak", 0.5)):
        synthetic.write_member(member, "P", "within", root, effect=effect)
    held_out = stats.compare("strong", "weak", "P", "within", root, sessions=["btbank5_0"], descriptive=True)
    assert held_out.decision == "Descriptive" and held_out.n_subjects == 1
    assert stats.best_baseline("P", "within", root, ("weak", "strong")) == "strong"
