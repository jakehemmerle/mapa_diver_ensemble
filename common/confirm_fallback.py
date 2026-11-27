"""Post-block amendment for the single frozen within-session uniform DIVER/MAPA design.

Constants come from the complete Tier P within regime, never from validation scores.
The source digests cover ordered (cell key, selected value, original configuration hash) rows.
"""
import hashlib
import json
import math

import numpy as np

PROTOCOL = 'within-single-class-select-v1'
FROZEN_SHA256 = 'f4bc9af1764bc6b655948676f89f5fe27d62e2cd242bb8a55ffbc33cfbf9a953'
SOURCE = {
    'mapa12': ('lam', 0.46415888336127775, '0fe3c653a9415792b10e64414db7ee7096690db0fb95026ea7ca9cd0752075d8'),
    'ridge6': ('lam', 2.154434690031882, 'ae911d53e814a39f41fb55cc9bd675c8deb0ac852bf747c8865ed36245a2ab94'),
    'diver': ('epochs', 8, 'd64009107058982c116e505978e0a13c4f475b5faae92e479f60a69a4c67b265'),
}


def in_scope(cell, member):
    return cell.tier in ('V', 'T') and cell.regime == 'within' and member in SOURCE


def for_select(cell, member, y_select):
    if not in_scope(cell, member):
        return None
    y = np.asarray(y_select)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError(f'{cell.key()}: empty, non-vector or nonfinite select labels; no fallback')
    if (y > 0).any() and (y <= 0).any():
        return None
    key, value, digest = SOURCE[member]
    return {'protocol': PROTOCOL, 'member': member, key: value, 'source_sha256': digest}


def verify_source(root):
    """Verify every contributing Tier P fit and recompute each preregistered median."""
    from common import predictions, ridge, splits
    cells = splits.list_cells('P', 'within')
    if len(cells) != 390:
        raise ValueError('the complete source protocol has 390 cells')
    for member, (key, value, expected_hash) in SOURCE.items():
        records, ranks = [], []
        for cell in cells:
            with np.load(predictions.path(cell, member, root), allow_pickle=False) as f:
                meta = json.loads(str(f['meta']))
            if (meta.get('seed', 0) != 0 or not math.isfinite(meta['select_auroc']) or
                    meta.get('status', 'ok') != 'ok' or meta.get('confirm_fallback') is not None):
                raise ValueError(f'{member} {cell.key()}: not a normal seed-zero fit')
            v = meta['best_epoch' if key == 'epochs' else 'lam']
            records.append([cell.key(), v, meta['config_hash']])
            ranks.append(v if key == 'epochs' else ridge.LAM_MULTS.index(v))
        digest = hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()
        mid = math.floor(np.median(ranks))  # indices/epochs are integers; half ties go to the lower integer
        if digest != expected_hash or (mid if key == 'epochs' else ridge.LAM_MULTS[mid]) != value:
            raise ValueError(f'{member}: pinned Tier P metadata or fallback parameter changed')
