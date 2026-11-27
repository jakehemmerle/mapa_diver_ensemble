import numpy as np
import pytest

from common import splits
from common.splits import Cell
from synthetic import Provider, make_table

RATE = 2048


@pytest.fixture
def provider():
    tables = {name: make_table(RATE, 600, seed=i) for i, name in enumerate(splits.EVAL_SESSIONS + splits.DEV_SESSIONS)}
    source = Provider(tables)
    splits.set_provider(source)
    return source


def test_cell_universe():
    assert len(splits.list_cells("T", "within")) == 12 * 15 * 2
    assert len(splits.list_cells("V", "cross_session")) == 12 * 15
    assert len(splits.list_cells("T", "cross_subject")) == 10 * 15
    assert len(splits.DEV_SESSIONS) == 13 and "btbank6_0" not in splits.DEV_SESSIONS
    assert len(splits.list_cells("P", "within")) == 13 * 15 * 2
    pairs = splits.pairs("P", "cross_session")
    assert len(pairs) == 22 and ("btbank6_1", "btbank6_4") in pairs and ("btbank6_4", "btbank6_1") in pairs
    assert all(a.split("_")[0] == b.split("_")[0] for a, b in pairs)
    targets = [target for anchor, target in splits.pairs("P", "cross_subject")]
    assert targets == ["btbank1_0", "btbank3_2", "btbank4_2", "btbank5_0", "btbank6_1", "btbank6_4",
                       "btbank8_0", "btbank9_0"]
    assert {anchor for anchor, _ in splits.pairs("P", "cross_subject")} == {"btbank2_6"}
    assert {anchor for anchor, _ in splits.pairs("T", "cross_subject")} == {"btbank2_4"}
    assert ("btbank1_2", "btbank1_1") in splits.pairs("T", "cross_session")
    cell = Cell("V", "within", "btbank3_0", "btbank3_0", "speech", 1)
    assert cell.key() == "btbank3_0__btbank3_0__speech__f1"


def test_mirror_matches_sklearn_kfold():
    from sklearn.model_selection import KFold
    for n in (10, 11, 3501):
        folds = splits.mirror_official("within", n, n)
        for fold, (train, test) in zip(folds, KFold(n_splits=2, shuffle=False).split(np.arange(n))):
            assert np.array_equal(fold["train"], train)
            assert np.array_equal(fold["val"], test[: len(test) // 2])
            assert np.array_equal(fold["test"], test[len(test) // 2:])


def test_tiers_do_not_cross(provider):
    with pytest.raises(ValueError):
        splits.get_cell(Cell("P", "within", "btbank3_0", "btbank3_0", "speech", 0))    # evaluation session in Tier P
    with pytest.raises(ValueError):
        splits.get_cell(Cell("V", "within", "btbank2_6", "btbank2_6", "speech", 0))    # permitted session in Tier V
    with pytest.raises(ValueError):
        splits.get_cell(Cell("P", "within", "btbank6_0", "btbank6_0", "speech", 0))    # excluded: evaluation film


@pytest.mark.parametrize("regime,train,target,fold", [
    ("within", "btbank3_0", "btbank3_0", 0), ("within", "btbank3_0", "btbank3_0", 1),
    ("cross_session", "btbank3_1", "btbank3_0", 0), ("cross_subject", "btbank2_4", "btbank3_0", 0)])
def test_tier_v_halves_official_val_with_a_purge(provider, regime, train, target, fold):
    official = provider.official(regime, train, target, "speech")[fold]
    table = provider.table(target, "speech")
    tier_t = splits.get_cell(Cell("T", regime, train, target, "speech", fold))
    tier_v = splits.get_cell(Cell("V", regime, train, target, "speech", fold))
    assert np.array_equal(tier_t.rows["select"], official["val"]) and np.array_equal(tier_t.rows["eval"], official["test"])
    assert np.array_equal(tier_v.rows["train"], official["train"])
    select, evaluate = tier_v.rows["select"], tier_v.rows["eval"]
    assert len(select) > 20 and len(evaluate) > 20
    assert not set(select) & set(evaluate)
    assert set(select) | set(evaluate) <= set(official["val"])          # no official test row is reachable
    assert not (set(select) | set(evaluate)) & set(official["test"])
    select_s, evaluate_s = table.onsets[select] / RATE, table.onsets[evaluate] / RATE
    assert select_s.max() < evaluate_s.min()                             # earlier half selects, later half scores
    assert evaluate_s.min() - select_s.max() >= 2 * splits.PURGE_S       # 2 s on each side of the cut
    dropped = np.setdiff1d(official["val"], np.concatenate([select, evaluate]))
    cut = 0.5 * (select_s.max() + evaluate_s.min())
    assert len(dropped) > 0 and np.all(np.abs(table.onsets[dropped] / RATE - cut) < splits.PURGE_S + 2.0)
    assert np.array_equal(tier_v.onsets("select"), table.onsets[select])
    assert np.array_equal(tier_v.labels("eval"), table.labels[evaluate])  # Tier V eval labels are val labels


def test_out_of_fold_blocks_are_purged_time_blocks(provider):
    split = splits.get_cell(Cell("T", "within", "btbank4_0", "btbank4_0", "pitch", 0))
    seconds = split.onsets("train") / RATE
    blocks = split.oof_blocks()
    assert len(blocks) == splits.N_OOF_BLOCKS
    held_all = np.concatenate([held for _, held in blocks])
    assert np.array_equal(np.sort(held_all), np.arange(len(seconds)))     # every training row is held out once
    previous_end = -np.inf
    for fit, held in sorted(blocks, key=lambda b: seconds[b[1]].min()):
        assert not set(fit) & set(held)
        low, high = seconds[held].min(), seconds[held].max()
        assert low > previous_end                                          # blocks are contiguous in time
        previous_end = high
        inside = (seconds[fit] >= low - splits.PURGE_S) & (seconds[fit] <= high + splits.PURGE_S)
        assert not inside.any()                                            # no fit window within 2 s of the block
        assert len(fit) < len(seconds) - len(held)                         # the purge removed neighbours


def test_tier_t_eval_labels_need_a_frozen_candidate(provider, tmp_path, monkeypatch):
    lock = pytest.importorskip("common.lock")
    monkeypatch.delenv("NPEXP_TEST_PURPOSE", raising=False)
    split = splits.get_cell(Cell("T", "cross_subject", "btbank2_4", "btbank7_0", "onset", 0))
    assert len(split.rows["eval"]) and len(split.onsets("eval"))          # rows and onsets are never gated
    assert len(split.labels("train")) and len(split.labels("select"))
    with pytest.raises(lock.TestLocked):
        split.labels("eval")
    tier_p = splits.get_cell(Cell("P", "cross_subject", "btbank2_6", "btbank9_0", "onset", 0))
    assert len(tier_p.labels("eval")) == len(tier_p.rows["eval"])
    assert tier_p.rates == {"train": 2048, "select": 1024, "eval": 1024}
