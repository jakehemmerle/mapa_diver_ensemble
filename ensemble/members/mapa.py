"""M1 and M2: closed-form ridge on MAPA kernels (last encoder layer, frontend).

Thin wrappers: kernels from ``common.mapa_features``, readout from ``common.ridge``.
The only quantity chosen on the select part is the ridge penalty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import confirm_fallback, ledger, mapa_features, predictions, ridge

TAP = {"mapa12": 12, "mapa0": 0}


def export_cell(member: str, cell, root: Path, *, splits=None) -> Path:
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    split = splits.get_cell(cell)
    y_select = split.labels("select")
    fallback = confirm_fallback.for_select(cell, member, y_select)
    kernel = mapa_features.kernels(cell, TAP[member], splits=splits)
    fit = ridge.fit_predict([kernel], split.labels("train"), y_select, oof_blocks=split.oof_blocks(),
                            fixed_lam=fallback["lam"] if fallback is not None else None,
                            strict=confirm_fallback.in_scope(cell, member))
    scores = {part: np.asarray(fit["scores"][part], dtype=np.float32) for part in predictions.PARTS}
    source = mapa_features.meta(cell, TAP[member])
    meta = {"commit": ledger.commit(), "tap": TAP[member], "lam": fit["lam"], "lam_pin": fit["lam_pin"],
            "n_tied": fit["n_tied"], "fit_select_auroc": fit["select_auroc"],
            # AUROC of the float32 scores as stored, by the member's own scorer
            "select_auroc": ridge.auroc(scores["select"], y_select) if fallback is None else None,
            "selected_on_select": ["ridge penalty (25-point grid, smallest on ties)"] if fallback is None else [],
            "kernel": {k: source.get(k) for k in ("feature", "n_columns", "flags", "mapa_revision",
                                                  "train_source", "target_source")},
            "config_hash": ledger.config_hash({"member": member, "tap": TAP[member], "grid": list(ridge.LAM_MULTS)})}
    if fallback is not None:
        meta["confirm_fallback"] = fallback
    return predictions.write(cell, member, split.rows, scores, meta, root)


def export(member: str, tier: str, regime: str, root: Path, *, splits=None, skip_existing: bool = True) -> int:
    """Write the member's predictions for every cell of a tier and regime. Reads training
    and select labels only; eval scores are computed without eval labels."""
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    written = 0
    for cell in splits.list_cells(tier, regime):
        if skip_existing and predictions.exists(cell, member, root):
            continue
        export_cell(member, cell, root, splits=splits)
        written += 1
    return written
