"""Read MAPA feature caches: ours (written by ``encode_session``) and the reproduction's.

Both expose feature rows by window onset (sample index in the session recording), so a
cell's rows are matched by onset, never by position.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNKNOWN_REGION_ID = 74      # MAPA's reserved id for contacts outside every atlas region
NATIVE_RATE = {9: 1024}     # Hz; every other Brain Treebank subject is 2048 (mapa.data.manifest)


def native_rate(subject: int) -> int:
    return NATIVE_RATE.get(int(subject), 2048)


class Features:
    """One session's features. ``column(tap, c)`` returns (windows, width) float16."""

    session: str
    onsets: np.ndarray            # (windows,) int64 sample index of each window start
    elec_labels: np.ndarray       # contact label per per-contact column
    parcels: np.ndarray           # atlas id per region column
    source: str
    flags: dict

    def positions(self, onsets) -> np.ndarray:
        onsets = np.asarray(onsets, dtype=np.int64)
        order = np.argsort(self.onsets, kind="stable")
        found = np.searchsorted(self.onsets[order], onsets)
        found[found == len(order)] = 0
        position = order[found]
        missing = self.onsets[position] != onsets
        if missing.any():
            raise KeyError(f"{self.source}: {int(missing.sum())} of {len(onsets)} window onsets have no "
                           f"encoded window (first {onsets[missing][:3].tolist()})")
        return position

    def has(self, tap: str) -> bool:
        raise NotImplementedError

    def column(self, tap: str, index: int) -> np.ndarray:
        raise NotImplementedError

    def n_columns(self, tap: str) -> int:
        return len(self.elec_labels if tap.endswith("_elec") else self.parcels)


class OwnFeatures(Features):
    """Directory written by ``ensemble.mapa.encode_session``. Region taps are stored
    (windows, regions, width); per-contact taps (contacts, windows, width) so that one
    contact is one contiguous read."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text())
        self.session = meta["session"]
        self.onsets = np.asarray(meta["onsets"], dtype=np.int64)
        self.elec_labels = np.asarray(meta["elec_labels"])
        self.parcels = np.asarray(meta["present_parcels"], dtype=np.int64)
        self.source = str(self.folder)
        self.flags = meta.get("flags", {})
        self.meta = meta
        self._maps: dict = {}

    def has(self, tap: str) -> bool:
        return (self.folder / f"{tap}.npy").is_file()

    def _map(self, tap: str):
        if tap not in self._maps:
            self._maps[tap] = np.load(self.folder / f"{tap}.npy", mmap_mode="r")
        return self._maps[tap]

    def column(self, tap: str, index: int) -> np.ndarray:
        data = self._map(tap)
        return np.ascontiguousarray(data[index] if tap.endswith("_elec") else data[:, index])


class ReproFeatures(Features):
    """``enc_s<S>_t<T>_<tag>.pt`` written by the reproduction's MAPA encode (read-only,
    memory-mapped). Only the label-free fields are used: features, window starts,
    contact labels and region ids."""

    def __init__(self, file: Path, session: str, subject: int):
        import torch
        self.file = Path(file)
        record = torch.load(self.file, map_location="cpu", weights_only=False, mmap=True)
        self.session = session
        self.onsets = np.rint(np.asarray(record["clip_starts"], dtype=np.float64) * native_rate(subject)).astype(np.int64)
        if len(np.unique(self.onsets)) != len(self.onsets):
            raise ValueError(f"{file}: window starts are not unique at the native sample rate")
        labels = record.get("elec_labels")
        self.elec_labels = np.asarray(labels) if labels is not None else np.array([], dtype=str)
        self.parcels = np.asarray(record["present_parcels"], dtype=np.int64)
        self.source = str(self.file)
        self.flags = {}
        self.checkpoint = record.get("checkpoint")
        self._feats = record["feats"]

    def has(self, tap: str) -> bool:
        return tap in self._feats

    def shape(self, tap: str) -> tuple:
        return tuple(self._feats[tap]["raw"].shape)

    def column(self, tap: str, index: int) -> np.ndarray:
        return np.ascontiguousarray(self._feats[tap]["raw"][:, index].numpy())


def shared_columns(train: Features, target: Features, tap: str) -> tuple[list[int], list[int], list]:
    """Aligned column indices, as MAPA's readout does: contacts by label for per-contact
    taps, regions by atlas id (never the 'unknown' id) for region taps."""
    if tap.endswith("_elec"):
        a, b = train.elec_labels, target.elec_labels
        common = np.intersect1d(a, b)
    else:
        a, b = train.parcels, target.parcels
        common = np.intersect1d(a, b)
        common = common[common != UNKNOWN_REGION_ID]
    if common.size == 0:
        raise ValueError(f"{train.session} and {target.session} share no {'contacts' if tap.endswith('_elec') else 'regions'}")
    return ([int(np.flatnonzero(a == c)[0]) for c in common],
            [int(np.flatnonzero(b == c)[0]) for c in common], common.tolist())
