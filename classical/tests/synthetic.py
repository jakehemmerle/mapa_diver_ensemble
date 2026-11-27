"""Synthetic sessions for the classical tests: benchmark-shaped tables and planted recordings."""
from __future__ import annotations

import numpy as np

from classical import features
from common import splits


def make_table(rate: int, n_events: int, seed: int, gap=(1.05, 1.6), lead_s: float = 2.0) -> splits.Table:
    """Rows as Neuroprobe orders them: classes interleaved, each class sorted in time."""
    rng = np.random.default_rng(seed)
    times = lead_s + np.cumsum(rng.uniform(*gap, size=n_events))
    event_labels = rng.integers(0, 2, size=n_events)
    positives, negatives = times[event_labels == 1], times[event_labels == 0]
    count = min(len(positives), len(negatives))
    onsets = np.empty(2 * count)
    onsets[0::2], onsets[1::2] = positives[:count], negatives[:count]   # row i has label (i + 1) % 2
    labels = (np.arange(2 * count) + 1) % 2
    return splits.Table(labels=labels.astype(np.int64), onsets=np.round(onsets * rate).astype(np.int64))


class Provider:
    """One table per session, shared by all tasks; official splits by the benchmark's arithmetic."""

    def __init__(self, tables: dict[str, splits.Table]) -> None:
        self.tables = tables

    def table(self, session: str, task: str) -> splits.Table:
        return self.tables[session]

    def official(self, regime: str, train_session: str, target_session: str, task: str):
        return splits.mirror_official(regime, len(self.tables[train_session].labels),
                                      len(self.tables[target_session].labels))


def evoked(rate: int, frequency: float = 6.0, decay_s: float = 0.3) -> np.ndarray:
    t = np.arange(rate) / rate
    return np.sin(2 * np.pi * frequency * t) * np.exp(-t / decay_s)


def make_recording(table: splits.Table, rate: int, seed: int, amplitude: float = 1.5,
                   regions: dict[str, str] | None = None) -> features.ArrayRecording:
    """Four contacts on shaft A, three on shaft B. Contact A2 carries a phase-locked response whose
    polarity is the label, so the two classes have identical magnitude spectra."""
    rng = np.random.default_rng(seed)
    n = int(table.onsets.max()) + 4 * rate
    data = {name: rng.standard_normal(n).astype(np.float32) for name in ("A1", "A2", "A3", "A4", "B1", "B2", "B3")}
    wave = evoked(rate) * amplitude
    for onset, label in zip(table.onsets, table.labels):
        data["A2"][onset:onset + rate] += (1.0 if label else -1.0) * wave
    regions = regions or {"A1": "ctx-lh-superiortemporal", "A2": "ctx-lh-middletemporal",
                          "A3": "ctx-lh-middletemporal", "A4": "Left-Cerebral-White-Matter",
                          "B1": "Left-Hippocampus", "B2": "Left-Hippocampus", "B3": "Unknown"}
    return features.ArrayRecording(rate=rate, data=data, regions=regions)
