"""Synthetic benchmark for tests and the server smoke run: tables, splits, predictions
and kernels with a known signal. Nothing here touches recordings or real labels."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from common import mapa_features, predictions, splits

ROWS = 240
SPACING = 3072          # samples between window starts (1.5 s at 2048 Hz)


def _rng(*parts) -> np.random.Generator:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class Provider:
    """``common.splits.Provider`` with alternating labels and evenly spaced onsets."""

    def __init__(self, rows: int = ROWS):
        self.rows = rows

    def table(self, session: str, task: str) -> splits.Table:
        index = np.arange(self.rows)
        rate = splits.native_rate(session)
        return splits.Table(labels=(index % 2).astype(np.int64),
                            onsets=(index * SPACING * rate // 2048).astype(np.int64))

    def official(self, regime, train_session, target_session, task):
        return splits.mirror_official(regime, self.rows, self.rows)


def install(rows: int = ROWS) -> Provider:
    provider = Provider(rows)
    splits.set_provider(provider)
    return provider


def member_scores(member: str, session: str, task: str, effect: float, rows: int = ROWS) -> np.ndarray:
    """Score of every dataset row: label times ``effect`` plus the member's own noise."""
    labels = np.arange(rows) % 2
    return effect * labels + _rng(member, session, task).normal(size=rows)


def write_member(member: str, tier: str, regime: str, root: Path, *, effect: float = 1.0,
                 parts=predictions.PARTS, noise_of: str | None = None) -> None:
    """Write a synthetic member for every cell. ``noise_of`` reuses another member's noise."""
    from common import stats
    for cell in splits.list_cells(tier, regime):
        split = splits.get_cell(cell)
        source = noise_of or member
        scores = {}
        for part in parts:
            session = cell.train_session if part == "train" else cell.target_session
            scores[part] = member_scores(source, session, cell.task, effect)[split.rows[part]].astype(np.float32)
        meta = {"synthetic": True, "select_auroc": stats.auroc(scores["select"], split.labels("select")),
                "commit": "test"}
        predictions.write(cell, member, split.rows, scores, meta, root)


def features(tap: int, session: str, task: str, rows: int = ROWS, width: int = 12) -> np.ndarray:
    """Features whose first columns carry the label; tap 12 is cleaner than tap 0."""
    labels = np.arange(rows) % 2
    x = _rng("features", tap, session, task).normal(size=(rows, width))
    x[:, :3] += (0.9 if tap == 12 else 0.5) * labels[:, None]
    return x


def write_kernels(tier: str, regime: str, work: Path) -> None:
    """Kernel files as the GPU job stores them, computed with numpy from ``features``."""
    store = mapa_features.STORE_OF_TIER[tier]
    for cell in splits.list_cells("P" if store == "P" else "T", regime):
        split = splits.get_cell(cell)
        target_rows = np.concatenate([split.rows["select"], split.rows["eval"]])
        for tap in mapa_features.TAPS:
            a = features(tap, cell.train_session, cell.task)[split.rows["train"]]
            b = features(tap, cell.target_session, cell.task)[target_rows]
            mean, scale = a.mean(0), a.std(0)
            scale[scale == 0] = 1
            a, b = (a - mean) / scale, (b - mean) / scale
            mapa_features.save(mapa_features.kernel_file(store, regime, cell.key(), tap, work),
                               train=a @ a.T, cross=b @ a.T, target=b @ b.T if tap == 0 else None,
                               train_rows=split.rows["train"], target_rows=target_rows,
                               meta={"synthetic": True, "feature": f"enc{tap}", "flags": {}})
