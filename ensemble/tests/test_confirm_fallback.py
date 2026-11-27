"""Integrity checks for the frozen within one-class amendment, synthetic data only."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from common import confirm_fallback as protocol, predictions, splits, stats
from ensemble import run


def test_missing_select_is_not_a_fallback():
    c = splits.Cell('V', 'within', 'btbank1_1', 'btbank1_1', 'onset', 0)
    for labels in ([], [np.nan], [[1]]):
        with pytest.raises(ValueError):
            protocol.for_select(c, 'mapa12', labels)
    assert protocol.for_select(c, 'mapa12', [0, 1]) is None
    assert protocol.for_select(dataclasses.replace(c, tier='P'), 'diver', [1, 1]) is None
    assert protocol.for_select(c, 'cnn', [1, 1]) is None


def test_one_class_integrity_rejects_fabricated_auc_wrong_parameters_and_undefined_cdf(root, monkeypatch):
    cell = splits.Cell('V', 'within', 'btbank1_1', 'btbank1_1', 'onset', 0)
    rows = {'select': np.arange(4), 'eval': np.arange(4, 8)}
    y = np.zeros(4)
    def labels(part):
        assert part == 'select'
        return y
    source = SimpleNamespace(list_cells=lambda *_: [cell], get_cell=lambda _: SimpleNamespace(rows=rows, labels=labels))
    monkeypatch.setattr(run, '_splits', lambda: source)
    monkeypatch.setattr(stats, '_splits', lambda _: source)
    scores = {p: np.arange(4.) for p in rows}
    receipt = protocol.for_select(cell, 'diver', y)
    m = dict(confirm_fallback=receipt, epochs_run=8, select_auroc=None)
    def verify(meta, arrays=scores):
        predictions.write(cell, 'diver', rows, arrays, meta, root)
        return run.check_member('diver','V','within',root,need_train=False)
    assert verify(m)['undefined_select_cells'] == [cell.key()]
    assert not verify(dict(m, epochs_run=7))['passed']
    assert not verify(dict(m, select_auroc=0.5))['passed']
    assert not verify(dict(m, confirm_fallback=None))['passed']
    assert not verify(m, dict(scores, select=np.ones(4)))['passed']
    assert not verify(m, {'select': scores['select']})['passed']
