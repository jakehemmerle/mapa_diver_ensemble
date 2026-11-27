"""Attest the original partial within V export before reusing it, without reading scoring-half labels."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import confirm_fallback, predictions, ridge, splits, stats

ORIGINAL = '93919c3eff96a7e6b43b4b9dc56b9c4f9d3c72d0'
BLOCKED_SHA256 = '398bce4af2ef54bf6d16c8de98fc8ce66f869d70e0e05694bf2a80e7dd51475a'
# Placement-only successor: the audit receipt was written by this release before any scoring attempt.
AUDITED_BY = 'c4bb6bf77a495385baeb86cb27cfe0e0e49f8767'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attest(root, commit):
    state = root / 'overnight'
    receipt = state / 'amendment_within_single_class_v1.json'
    frozen_sha = sha256(root / 'frozen' / 'within.json')
    if frozen_sha != confirm_fallback.FROZEN_SHA256 or sha256(state / 'confirmation_blocked_within.json') != BLOCKED_SHA256:
        raise ValueError('original frozen design or blocked preparation receipt changed')
    confirm_fallback.verify_source(root)
    namespace = root / 'amendments' / confirm_fallback.PROTOCOL
    # On a resumed *preparation*, verify the original file bytes are still unchanged.
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if recorded['commit'] not in (commit, AUDITED_BY) or recorded['protocol'] != confirm_fallback.PROTOCOL:
            raise ValueError('amendment release identity differs')
        for name, digest in recorded['original_files'].items():
            if sha256(root / 'preds/V/within/mapa12' / name) != digest:
                raise ValueError(f'original prediction changed: {name}')
            if sha256(namespace / 'preds/V/within/mapa12' / name) != digest:
                raise ValueError(f'amended copy of original prediction changed: {name}')
        return recorded
    if list(state.glob('attempt_*_within.json')):
        raise ValueError('cannot amend after a held-out scoring attempt')
    cells = splits.list_cells('V', 'within')
    present = sorted((root / 'preds/V/within').glob('*/*.npz'))
    expected = [root / 'preds/V/within/mapa12' / f'{c.key()}.npz' for c in cells[:65]]
    if namespace.exists():
        raise ValueError('amended output namespace already exists without its audit receipt')
    if len(cells) != 360 or present != sorted(expected):
        raise ValueError('original partial export is not the exact 65 MAPA12 files at the block')
    original_files = {}
    for cell, path in zip(cells, expected):
        p = predictions.read_file(path)
        s = splits.get_cell(cell)
        y = s.labels('select')  # the veto half is never read
        if (p.meta['commit'] != ORIGINAL or p.meta['config_hash'] != '94814faa0f6a1d56' or p.meta['tap'] != 12 or
                p.meta['lam'] not in ridge.LAM_MULTS or set(p.scores) != set(splits.PARTS) or
                confirm_fallback.for_select(cell, 'mapa12', y) is not None):
            raise ValueError(f'{cell.key()}: not an ordinary fit from the original release')
        for part in splits.PARTS:
            if not np.array_equal(p.rows[part], s.rows[part]) or not np.isfinite(p.scores[part]).all():
                raise ValueError(f'{cell.key()}/{part}: invalid rows or scores')
        gap = abs(stats.auroc(p.scores['select'], y) - p.meta['select_auroc'])
        if not gap <= 1e-6 or len(np.unique(p.scores['select'])) < 2:
            raise ValueError(f'{cell.key()}: invalid select scores')
        original_files[path.name] = sha256(path)
    namespace.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=namespace.parent) as staging:
        staged = Path(staging) / 'amendment'
        target = staged / 'preds/V/within/mapa12'
        target.mkdir(parents=True)
        for path in expected:
            copy = target / path.name
            shutil.copyfile(path, copy)
            if sha256(copy) != original_files[path.name]:
                raise ValueError(f'audited copy differs: {path.name}')
        staged.rename(namespace)
    result = {'protocol': confirm_fallback.PROTOCOL, 'commit': commit, 'prior_release': ORIGINAL,
              'frozen_sha256': frozen_sha, 'original_blocked_sha256': BLOCKED_SHA256,
              'tier_p_parameters': confirm_fallback.SOURCE, 'original_files': original_files,
              'prediction_namespace': str(namespace.relative_to(root)),
              'gate': {'minimum_delta': 0.005, 'minimum_subjects_up': 5, 'n_subjects': 6}}
    with receipt.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result
