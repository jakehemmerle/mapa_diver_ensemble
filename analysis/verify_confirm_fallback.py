"""Practice the amended exporters and integrity check on one permitted cell. No V/T data.

The simulated single-class arm replaces only select labels with zeros and activates the
confirmation predicate for those labels. Actual data loading, kernels, heads, fitters,
CDF fusion and select integrity checks are unchanged; only private output paths are used.
"""
import dataclasses
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from classical import run as classical
from common import confirm_fallback, ledger, predictions, splits, stats
from ensemble import fuse, paths, run
from ensemble.members import diver, mapa, trained


def main():
    production = ledger.results_root()
    confirm_fallback.verify_source(production)
    cell, = (c for c in splits.list_cells('P', 'within') if c.target_session == 'btbank1_0' and c.task == 'onset' and c.fold == 0)
    in_scope = confirm_fallback.in_scope
    original = splits.get_cell(cell)
    decision = confirm_fallback.for_select
    # This token only exercises the predicate. It is never passed to a data provider.
    token = dataclasses.replace(cell, tier='V')
    evidence = {}
    for forced in (False, True):
        def checked_get(c):
            if c != cell:
                raise ValueError('practice cannot access any other data')
            return practice

        def checked_labels(part):
            if part not in ('train', 'select'):
                raise ValueError('practice fit attempted to read outcome labels')
            y = original.labels(part)
            return np.zeros_like(y) if forced and part == 'select' else y

        practice = dataclasses.replace(original, _labels={p: (lambda p=p: checked_labels(p)) for p in splits.PARTS})
        with tempfile.TemporaryDirectory(dir=paths.scratch()) as folder, \
                patch.object(splits, 'get_cell', checked_get), patch.object(splits, 'list_cells', return_value=[cell]), \
                patch.object(confirm_fallback, 'in_scope', side_effect=lambda c,m: in_scope(token,m)), \
                patch.object(confirm_fallback, 'for_select', side_effect=lambda c,m,y: decision(token,m,y)):
            out = Path(folder)
            mapa.export_cell('mapa12', cell, out)
            records = classical.run_cell((cell, [classical.base_arm()], str(out), 'base6', False))
            if any(r['status'] not in ('ok', 'existing') for r in records):
                raise ValueError(f'practice ridge export failed: {records}')
            assert trained.export(diver.Backend(), 'P', 'within', out, seeds=(0,), oof=False) == 1
            stats._label_cache.clear()
            for m in ('mapa12', 'ridge6', 'diver'):
                p = predictions.read(cell, m, out)
                check = run.check_member(m, 'P', 'within', out, need_train=m != 'diver')
                assert check['passed'], check
                if forced:
                    assert check['undefined_select_cells'] == [cell.key()]
                    assert p.meta['select_auroc'] is None
                    assert p.meta['confirm_fallback'] == decision(token, m, checked_labels('select'))
                    assert np.unique(p.scores['select']).size > 1
                    evidence['fixed_' + m] = p.meta['confirm_fallback']
                else:
                    ref = predictions.read(cell, m, production)
                    par = 'best_epoch' if m == 'diver' else 'lam'
                    assert p.meta[par] == ref.meta[par] and p.meta['config_hash'] == ref.meta['config_hash']
                    gaps = {}
                    for part in ('select', 'eval'):
                        assert np.array_equal(p.rows[part], ref.rows[part])
                        gaps[part] = float(np.max(np.abs(p.scores[part] - ref.scores[part])))
                    assert max(gaps.values()) <= 1e-6, (m, gaps)
                    evidence['ordinary_' + m] = gaps
            name = fuse.fuse(['diver','mapa12'], 'a', 'P', 'within', out)
            p = predictions.read(cell, name, out)
            assert all(np.array_equal(p.rows[k], original.rows[k]) and np.isfinite(p.scores[k]).all() for k in ('select','eval'))
            if not forced:
                ref = predictions.read(cell, name, production)
                gaps = {part: float(np.max(np.abs(p.scores[part] - ref.scores[part]))) for part in ('select','eval')}
                assert max(gaps.values()) <= 1e-6, gaps
                evidence['ordinary_fused'] = gaps
    print(json.dumps({'passed': True, 'cell': cell.key(), 'commit': ledger.commit(), 'evidence': evidence}, sort_keys=True))


if __name__ == '__main__':
    main()
