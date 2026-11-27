"""Evaluation cells and their row splits for Tiers P, V and T.

Rows are indices into Neuroprobe's own benchmark dataset of ``(session, task)``
(``BrainTreebankSubjectTrialBenchmarkDataset``, Lite electrodes, revision f9b0842).
The official train/val/test rows always come from Neuroprobe's split generators;
this module only renames and, for Tier V, subdivides them:

====  =============================  ==========================================
Tier  select                         eval
====  =============================  ==========================================
P     official val                   official test (13 permitted sessions)
V     earlier half of official val   later half of official val, windows starting
                                     within 2 s of the cut dropped on both sides
T     official val                   official test; ``labels("eval")`` goes
                                     through ``common.lock.require_test_unlocked``
====  =============================  ==========================================

Tier V halves are taken in onset-time order (the benchmark interleaves classes, so
row order is not time order) which is what makes the 2 s purge meaningful.

Tier P sessions are registered in-process by appending to
``NEUROPROBE_LITE_SUBJECT_TRIALS`` (the dataset asserts membership); Lite
electrodes exist for all 10 subjects. Nothing in the installed package is edited.

Row sets and onsets are never gated (features may be computed for every row);
only Tier T evaluation labels are.
"""
from __future__ import annotations

import contextlib
import functools
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

import common

TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index", "word_gap", "gpt2_surprisal",
    "word_head_pos", "word_part_speech", "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
REGIMES = ("within", "cross_session", "cross_subject")
TIERS = ("P", "V", "T")
EVAL_SESSIONS = ("btbank1_1", "btbank1_2", "btbank2_0", "btbank2_4", "btbank3_0", "btbank3_1",
                 "btbank4_0", "btbank4_1", "btbank7_0", "btbank7_1", "btbank10_0", "btbank10_1")
# Permitted sessions whose film no evaluation session shows (plan, "Evaluation tiers").
DEV_SESSIONS = ("btbank1_0", "btbank2_1", "btbank2_2", "btbank2_3", "btbank2_5", "btbank2_6", "btbank3_2",
                "btbank4_2", "btbank5_0", "btbank6_1", "btbank6_4", "btbank8_0", "btbank9_0")
EVAL_ANCHOR = "btbank2_4"   # neuroprobe/config.py DS_DM_TRAIN_*
DEV_ANCHOR = "btbank2_6"
PURGE_S = 2.0
N_OOF_BLOCKS = 5
WINDOW_S = 1.0
PARTS = ("train", "select", "eval")


class MissingInput(RuntimeError):
    """A file a cell needs is not on disk yet (for example a Tier P recording)."""


def parse_session(name: str) -> tuple[int, int]:
    subject, trial = name.removeprefix("btbank").split("_")
    return int(subject), int(trial)


def session_name(subject: int, trial: int) -> str:
    return f"btbank{subject}_{trial}"


def native_rate(session: str) -> int:
    """Sampling rate of the recording. Subject 9 is distributed at 1024 Hz, the rest at 2048 Hz."""
    return 1024 if parse_session(session)[0] == 9 else 2048


@dataclass(frozen=True)
class Cell:
    tier: str            # "P" | "V" | "T"
    regime: str          # "within" | "cross_session" | "cross_subject"
    train_session: str   # e.g. "btbank2_4"
    target_session: str
    task: str            # Neuroprobe task key, e.g. "word_index"
    fold: int

    def key(self) -> str:
        return f"{self.target_session}__{self.train_session}__{self.task}__f{self.fold}"


# --------------------------------------------------------------------------- providers

@dataclass(frozen=True)
class Table:
    """All rows of one benchmark dataset: label and window-start sample of each row."""
    labels: np.ndarray   # (n,) int
    onsets: np.ndarray   # (n,) int64, sample index into the session recording


class Provider(Protocol):
    def table(self, session: str, task: str) -> Table: ...
    def official(self, regime: str, train_session: str, target_session: str, task: str) -> list[dict[str, np.ndarray]]:
        """Per fold ``{"train": rows of train_session, "val": ..., "test": rows of target_session}``."""


def mirror_official(regime: str, n_train: int, n_target: int) -> list[dict[str, np.ndarray]]:
    """The benchmark's split arithmetic on row counts alone (train_test_splits.py at f9b0842).

    Used to cross-check what the real generators return and by synthetic providers in tests.
    """
    def halves(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rows[: len(rows) // 2], rows[len(rows) // 2:]

    if regime == "within":
        assert n_train == n_target
        first, second = np.arange(n_target)[: (n_target + 1) // 2], np.arange(n_target)[(n_target + 1) // 2:]
        folds = []
        for held, train in ((first, second), (second, first)):   # KFold(2, shuffle=False)
            val, test = halves(held)
            folds.append({"train": train, "val": val, "test": test})
        return folds
    val, test = halves(np.arange(n_target))
    return [{"train": np.arange(n_train), "val": val, "test": test}]


class NeuroprobeProvider:
    """Tables and official splits from the installed, pinned Neuroprobe package."""

    def __init__(self) -> None:
        os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", str(common.data_root()))
        import neuroprobe.config as config
        import neuroprobe.train_test_splits as generators
        for name in DEV_SESSIONS:   # in place: datasets.py and train_test_splits.py share this list object
            if parse_session(name) not in config.NEUROPROBE_LITE_SUBJECT_TRIALS:
                config.NEUROPROBE_LITE_SUBJECT_TRIALS.append(parse_session(name))
        self._config, self._generators = config, generators
        self._subjects: dict[int, object] = {}
        self._tables: dict[tuple[str, str], Table] = {}
        self._official: dict[tuple[str, str, str, str], list[dict[str, np.ndarray]]] = {}

    def subject(self, subject_id: int):
        if subject_id not in self._subjects:
            from neuroprobe import BrainTreebankSubject
            self._subjects[subject_id] = BrainTreebankSubject(subject_id, cache=False)
        return self._subjects[subject_id]

    def require_recording(self, session: str) -> None:
        subject, trial = parse_session(session)
        path = common.data_root() / f"sub_{subject}_trial{trial:03d}.h5"
        if not path.exists():
            raise MissingInput(f"recording not on disk: {path}")

    def dataset(self, session: str, task: str):
        import torch
        from neuroprobe import BrainTreebankSubjectTrialBenchmarkDataset
        self.require_recording(session)
        subject, trial = parse_session(session)
        return BrainTreebankSubjectTrialBenchmarkDataset(
            self.subject(subject), trial, dtype=torch.float32, eval_name=task, output_indices=True, lite=True)

    def _remember(self, dataset) -> Table:
        key = (session_name(dataset.subject_id, dataset.trial_id), dataset.eval_name)
        if key not in self._tables:
            items = [dataset[i] for i in range(len(dataset))]
            self._tables[key] = Table(labels=np.array([item["label"] for item in items], dtype=np.int64),
                                      onsets=np.array([item["data"][0] for item in items], dtype=np.int64))
        return self._tables[key]

    def table(self, session: str, task: str) -> Table:
        if (session, task) not in self._tables:
            self._remember(self.dataset(session, task))
        return self._tables[(session, task)]

    @contextlib.contextmanager
    def _patched(self, **values):
        saved = {name: getattr(self._generators, name) for name in values}
        try:
            for name, value in values.items():
                setattr(self._generators, name, value)
            yield
        finally:
            for name, value in saved.items():
                setattr(self._generators, name, value)

    def official(self, regime: str, train_session: str, target_session: str, task: str) -> list[dict[str, np.ndarray]]:
        key = (regime, train_session, target_session, task)
        if key in self._official:
            return self._official[key]
        for session in {train_session, target_session}:
            self.require_recording(session)
        generators = self._generators
        (train_subject, train_trial), (subject, trial) = parse_session(train_session), parse_session(target_session)
        options = dict(output_indices=True, lite=True)
        if regime == "within":
            splits = generators.generate_splits_within_session(self.subject(subject), trial, task, **options)
        elif regime == "cross_session":
            # The generator trains on "the other Lite trial of the subject"; name that trial explicitly.
            with self._patched(NEUROPROBE_LITE_SUBJECT_TRIALS=[(subject, train_trial), (subject, trial)]):
                splits = generators.generate_splits_cross_session(self.subject(subject), trial, task, **options)
        elif regime == "cross_subject":
            subjects = {train_subject: self.subject(train_subject), subject: self.subject(subject)}
            with self._patched(DS_DM_TRAIN_SUBJECT_ID=train_subject, DS_DM_TRAIN_TRIAL_ID=train_trial):
                splits = generators.generate_splits_cross_subject(subjects, subject, trial, task, **options)
        else:
            raise ValueError(f"unknown regime {regime!r}")
        folds = []
        for split in splits:
            fold = {}
            for part in ("train", "val", "test"):
                rows, base = _unwrap(split[f"{part}_dataset"])
                expected = train_session if part == "train" else target_session
                if session_name(base.subject_id, base.trial_id) != expected or base.eval_name != task:
                    raise RuntimeError(f"{key}: generator returned {base.subject_id}_{base.trial_id} for {part}")
                self._remember(base)
                fold[part] = rows
            folds.append(fold)
        mirror = mirror_official(regime, len(self.table(train_session, task).labels),
                                 len(self.table(target_session, task).labels))
        if len(mirror) != len(folds) or any(not np.array_equal(a[p], b[p]) for a, b in zip(folds, mirror) for p in a):
            raise RuntimeError(f"{key}: Neuroprobe's split differs from the arithmetic this code documents")
        self._official[key] = folds
        return folds


def _unwrap(dataset) -> tuple[np.ndarray, object]:
    """Row indices of a (possibly nested) torch Subset in its base dataset, and that dataset."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        rows, base = _unwrap(dataset.dataset)
        return rows[np.asarray(dataset.indices, dtype=np.int64)], base
    return np.arange(len(dataset), dtype=np.int64), dataset


_provider: Provider | None = None


def set_provider(provider: Provider | None) -> None:
    """Install a table/split source (tests use synthetic ones); ``None`` restores Neuroprobe."""
    global _provider
    _provider = provider
    get_cell.cache_clear()


def provider() -> Provider:
    global _provider
    if _provider is None:
        _provider = NeuroprobeProvider()
    return _provider


def dataset(session: str, task: str):
    """Neuroprobe's benchmark dataset object for ``(session, task)`` (``output_indices=True``)."""
    source = provider()
    if not isinstance(source, NeuroprobeProvider):
        raise RuntimeError("dataset() needs the Neuroprobe provider")
    return source.dataset(session, task)


# --------------------------------------------------------------------------- cells

def _sessions(tier: str) -> tuple[str, ...]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return DEV_SESSIONS if tier == "P" else EVAL_SESSIONS


def pairs(tier: str, regime: str) -> list[tuple[str, str]]:
    """Ordered ``(train_session, target_session)`` pairs of a tier and regime."""
    sessions = _sessions(tier)
    if regime == "within":
        return [(s, s) for s in sessions]
    if regime == "cross_session":
        # Tier T/V: the subject's other evaluation session. Tier P: every ordered pair inside
        # subjects 2 and 6, the only subjects with two permitted sessions.
        return [(a, b) for b in sessions for a in sessions
                if a != b and parse_session(a)[0] == parse_session(b)[0]]
    if regime == "cross_subject":
        anchor = DEV_ANCHOR if tier == "P" else EVAL_ANCHOR
        return [(anchor, s) for s in sessions if parse_session(s)[0] != parse_session(anchor)[0]]
    raise ValueError(f"unknown regime {regime!r}")


def list_cells(tier: str, regime: str) -> list[Cell]:
    folds = (0, 1) if regime == "within" else (0,)
    return [Cell(tier, regime, train, target, task, fold)
            for train, target in pairs(tier, regime) for task in TASKS for fold in folds]


def tier_v_halves(val_rows: np.ndarray, onsets: np.ndarray, rate: int,
                  purge_s: float = PURGE_S) -> tuple[np.ndarray, np.ndarray]:
    """Split official val rows into (select, eval) by onset time with a purge around the cut."""
    seconds = onsets[val_rows] / rate
    order = np.argsort(seconds, kind="stable")
    half = len(order) // 2
    if half == 0:
        return val_rows[:0], val_rows[:0]
    cut = 0.5 * (seconds[order[half - 1]] + seconds[order[half]])
    early, late = order[:half], order[half:]
    select = val_rows[early[seconds[early] <= cut - purge_s]]
    evaluate = val_rows[late[seconds[late] >= cut + purge_s]]
    return np.sort(select), np.sort(evaluate)


@dataclass
class CellSplit:
    cell: Cell
    rows: dict[str, np.ndarray]                     # int64 dataset rows: "train" in train_session, others in target
    rates: dict[str, int]                           # native sampling rate per part
    _onsets: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    _labels: dict[str, Callable[[], np.ndarray]] = field(repr=False, default_factory=dict)

    def session(self, part: str) -> str:
        return self.cell.train_session if part == "train" else self.cell.target_session

    def onsets(self, part: str) -> np.ndarray:
        """Sample index of each window start in the session recording (native rate, as the benchmark slices it)."""
        return self._onsets[part]

    def labels(self, part: str) -> np.ndarray:
        if part not in PARTS:
            raise KeyError(part)
        if self.cell.tier == "T" and part == "eval":
            from common import lock   # fails closed if the lock module is absent
            lock.require_test_unlocked(self.cell.tier, self.cell.regime, common.results_root())
        return self._labels[part]()

    def oof_blocks(self, n_blocks: int = N_OOF_BLOCKS, purge_s: float = PURGE_S) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(fit_pos, held_pos)`` pairs of positions into ``rows["train"]``.

        Held blocks are contiguous in onset time and partition the training rows. A fit set
        omits the held block and every row whose window starts within ``purge_s`` of it.
        """
        seconds = self._onsets["train"] / self.rates["train"]
        order = np.argsort(seconds, kind="stable")
        blocks = []
        for held in np.array_split(order, n_blocks):
            if len(held) == 0:
                continue
            low, high = seconds[held].min(), seconds[held].max()
            keep = (seconds < low - purge_s) | (seconds > high + purge_s)
            blocks.append((np.flatnonzero(keep), np.sort(held)))
        return blocks


@functools.lru_cache(maxsize=4096)
def get_cell(cell: Cell) -> CellSplit:
    if (cell.train_session, cell.target_session) not in pairs(cell.tier, cell.regime):
        raise ValueError(f"{cell} is not a cell of tier {cell.tier} ({cell.regime})")
    if cell.task not in TASKS:
        raise ValueError(f"unknown task {cell.task!r}")
    source = provider()
    folds = source.official(cell.regime, cell.train_session, cell.target_session, cell.task)
    if not 0 <= cell.fold < len(folds):
        raise ValueError(f"{cell}: fold out of range ({len(folds)} folds)")
    official = folds[cell.fold]
    train_table = source.table(cell.train_session, cell.task)
    target_table = source.table(cell.target_session, cell.task)
    rates = {"train": native_rate(cell.train_session), "select": native_rate(cell.target_session),
             "eval": native_rate(cell.target_session)}
    if cell.tier == "V":
        select, evaluate = tier_v_halves(official["val"], target_table.onsets, rates["select"])
    else:
        select, evaluate = official["val"], official["test"]
    rows = {"train": np.asarray(official["train"], dtype=np.int64), "select": np.asarray(select, dtype=np.int64),
            "eval": np.asarray(evaluate, dtype=np.int64)}
    tables = {"train": train_table, "select": target_table, "eval": target_table}
    return CellSplit(
        cell=cell, rows=rows, rates=rates,
        _onsets={part: tables[part].onsets[rows[part]] for part in PARTS},
        _labels={part: (lambda part=part: tables[part].labels[rows[part]]) for part in PARTS},
    )
