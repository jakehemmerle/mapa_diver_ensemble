"""Paired subject-level comparison of two members, with the plan's threshold table.

Unit of analysis is the subject of the target session: AUROC is averaged over folds,
then sessions, then tasks within a subject. The test is an exact one-sided sign-flip
permutation over subjects; the interval is a session-level paired bootstrap; per-task
p-values are Holm-corrected over the tasks present.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from common import predictions

# (tier, regime) -> Go threshold, subjects that must be up, expected subjects, Hold floor.
THRESHOLDS = {
    ("P", "within"): {"go": 0.010, "up": 7, "of": 8, "hold": 0.005},
    ("P", "cross_subject"): {"go": 0.015, "up": 6, "of": 7, "hold": 0.008},
    ("V", "within"): {"go": 0.005, "up": 5, "of": 6},
    ("V", "cross_session"): {"go": 0.005, "up": 5, "of": 6},
    ("V", "cross_subject"): {"go": 0.010, "up": 4, "of": 5},
}
_EPS = 1e-12


def auroc(scores, labels) -> float:
    """Area under the ROC curve with tied scores at half credit; NaN if one class is absent.
    Positives are ``labels > 0``, as in MAPA's readout."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel() > 0
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], s.size]
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sign_flip_p(deltas) -> float:
    """Exact one-sided p-value: the share of all 2^n sign assignments whose mean is at
    least the observed mean."""
    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    if d.size > 20:
        raise ValueError("exact enumeration is limited to 20 units")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=d.size)))
    means = signs @ d / d.size
    return float(np.mean(means >= d.mean() - _EPS))


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values for a family given as ``{name: p}``."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def decide(tier: str, regime: str, delta: float, subjects_up: int, n_subjects: int) -> str:
    """Decision from the plan's table. Tier P: Go / Hold / Stop (a delta above the Go
    threshold with too few subjects up is Hold). Tier V: Go / Veto. Tier P cross-session
    is Descriptive; Tier T is Reported."""
    if tier == "T":
        return "Reported"
    rule = THRESHOLDS.get((tier, regime))
    if rule is None:
        return "Descriptive"
    if n_subjects != rule["of"]:
        raise ValueError(f"{tier}/{regime} expects {rule['of']} subjects, the comparison has {n_subjects}")
    go = delta >= rule["go"] - _EPS and subjects_up >= rule["up"]
    if tier == "V":
        return "Go" if go else "Veto"
    if go:
        return "Go"
    return "Hold" if delta >= rule["hold"] - _EPS else "Stop"


def subject_of(session: str) -> int:
    match = re.fullmatch(r"btbank(\d+)_(\d+)", session)
    if not match:
        raise ValueError(f"unrecognized session {session!r}")
    return int(match.group(1))


@dataclass
class Comparison:
    candidate: str
    baseline: str
    tier: str
    regime: str
    score: float
    baseline_score: float
    delta: float
    n_subjects: int
    subjects_up: int
    p_signflip: float
    ci_lo: float
    ci_hi: float
    decision: str
    per_subject: dict = field(default_factory=dict)   # subject -> delta
    per_session: dict = field(default_factory=dict)   # session -> delta
    per_task: dict = field(default_factory=dict)      # task -> {delta, p, p_holm}
    leave_one_subject_out: dict = field(default_factory=dict)  # subject removed -> delta
    n_cells: int = 0
    n_nan_cells: int = 0
    sessions: tuple = ()

    def row(self, phase: str, kind: str = "comparison", **extra) -> dict:
        """Ledger row for this comparison."""
        body = asdict(self)
        row = {"tier": self.tier, "regime": self.regime, "phase": phase, "kind": kind,
               "arm": self.candidate, "baseline": self.baseline, "score": self.score,
               "baseline_score": self.baseline_score, "delta": self.delta,
               "n_subjects": self.n_subjects, "subjects_up": self.subjects_up,
               "p_signflip": self.p_signflip, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
               "decision": self.decision,
               "extra": {k: body[k] for k in ("per_subject", "per_session", "per_task",
                                              "leave_one_subject_out", "n_cells", "n_nan_cells")}}
        row["extra"].update(extra.pop("extra", {}))
        row.update(extra)
        return row


def _splits(splits):
    if splits is None:
        from common import splits as module
        return module
    return splits


_label_cache: dict = {}


def _part_labels(splits, cell, part: str):
    key = (id(splits), cell, part)
    if key not in _label_cache:
        split = splits.get_cell(cell)
        # The splits object is kept in the entry so its id cannot be reused by another one.
        _label_cache[key] = (splits, np.asarray(split.rows[part], dtype=np.int64), np.asarray(split.labels(part)))
    return _label_cache[key][1:]


def cell_auroc(member: str, cell, root: Path, *, part: str = "eval", splits=None) -> float:
    """AUROC of one member on one part of one cell, labels aligned by row index."""
    rows, labels = _part_labels(_splits(splits), cell, part)
    pred = predictions.read(cell, member, root)
    if part not in pred.scores:
        raise KeyError(f"{member} has no {part} scores for {cell.key()}")
    scores = predictions.align(pred.scores[part], pred.rows[part], rows, f"{member} vs split, {cell.key()} {part}")
    return auroc(scores, labels)


def _hierarchy(values: dict) -> tuple[dict, dict, dict]:
    """``{(session, task, fold): x}`` -> per-subject, per-session and per-(subject, task) means."""
    by_fold: dict = {}
    for (session, task, _fold), x in values.items():
        by_fold.setdefault((session, task), []).append(x)
    session_task = {k: float(np.mean(v)) for k, v in by_fold.items()}
    by_session: dict = {}
    by_subject_task: dict = {}
    for (session, task), x in session_task.items():
        by_session.setdefault(session, []).append(x)
        by_subject_task.setdefault((subject_of(session), task), []).append(x)
    subject_task = {k: float(np.mean(v)) for k, v in by_subject_task.items()}
    by_subject: dict = {}
    for (subject, _task), x in subject_task.items():
        by_subject.setdefault(subject, []).append(x)
    return ({s: float(np.mean(v)) for s, v in by_subject.items()},
            {s: float(np.mean(v)) for s, v in by_session.items()}, subject_task)


def member_scores(member: str, tier: str, regime: str, root: Path, *, part: str = "eval",
                  splits=None, sessions=None) -> dict:
    """``{(target_session, task, fold): AUROC}`` over the cells of a tier and regime."""
    out = {}
    for cell in _splits(splits).list_cells(tier, regime):
        if sessions is not None and cell.target_session not in sessions:
            continue
        out[(cell.target_session, cell.task, cell.fold)] = cell_auroc(member, cell, root, part=part, splits=splits)
    return out


def macro(member: str, tier: str, regime: str, root: Path, *, part: str = "eval", splits=None,
          sessions=None) -> float:
    """Subject-level mean AUROC (folds, then sessions, then tasks, then subjects)."""
    values = {k: v for k, v in member_scores(member, tier, regime, root, part=part, splits=splits,
                                             sessions=sessions).items() if not np.isnan(v)}
    if not values:
        raise ValueError(f"{member}: no scored cells in {tier}/{regime}")
    return float(np.mean(list(_hierarchy(values)[0].values())))


def best_baseline(tier: str, regime: str, root: Path, members=("mapa12", "ridge6"), *,
                  splits=None, sessions=None) -> str:
    """The better of reproduced MAPA and reproduced Ridge 6-band in this tier and regime."""
    scored = {m: macro(m, tier, regime, root, splits=splits, sessions=sessions) for m in members}
    return max(members, key=lambda m: scored[m])


def compare(candidate: str, baseline: str, tier: str, regime: str, root: Path, *, splits=None,
            sessions=None, descriptive: bool = False, n_boot: int = 10000, seed: int = 0) -> "Comparison":
    """Compare two members on the eval part of every cell of ``tier``/``regime``.

    Both members must have scored exactly the split's eval rows for every cell
    (:class:`common.predictions.RowSetMismatch` otherwise). ``sessions`` restricts the
    target sessions; such sub-analyses must pass ``descriptive=True`` and get no decision.
    """
    splits = _splits(splits)
    cand, base, n_nan = {}, {}, 0
    cells = [c for c in splits.list_cells(tier, regime) if sessions is None or c.target_session in sessions]
    if not cells:
        raise ValueError(f"no cells for {tier}/{regime}")
    for cell in cells:
        rows, labels = _part_labels(splits, cell, "eval")
        pc = predictions.read(cell, candidate, root)
        pb = predictions.read(cell, baseline, root)
        predictions.align(pb.scores["eval"], pb.rows["eval"], pc.rows["eval"],
                          f"{baseline} vs {candidate}, {cell.key()} eval")
        a = auroc(predictions.align(pc.scores["eval"], pc.rows["eval"], rows, f"{candidate} vs split, {cell.key()}"), labels)
        b = auroc(predictions.align(pb.scores["eval"], pb.rows["eval"], rows, f"{baseline} vs split, {cell.key()}"), labels)
        if np.isnan(a) or np.isnan(b):
            n_nan += 1
            continue
        key = (cell.target_session, cell.task, cell.fold)
        cand[key], base[key] = a, b
    if not cand:
        raise ValueError(f"every cell of {tier}/{regime} has a single-class eval part")
    delta_cells = {k: cand[k] - base[k] for k in cand}
    subj_c, _, _ = _hierarchy(cand)
    subj_b, _, _ = _hierarchy(base)
    subj_d, sess_d, subj_task_d = _hierarchy(delta_cells)
    subjects = sorted(subj_d)
    deltas = np.array([subj_d[s] for s in subjects])
    delta = float(deltas.mean())
    up = int((deltas > 0).sum())

    # Session-level paired bootstrap of the subject-level mean delta.
    session_names = sorted(sess_d)
    session_delta = np.array([sess_d[s] for s in session_names])
    membership = np.array([[subject_of(s) == subject for s in session_names] for subject in subjects], dtype=float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(session_names), np.full(len(session_names), 1 / len(session_names)), size=n_boot)
    weight = counts @ membership.T                                   # (B, subjects): sessions drawn
    total = (counts * session_delta) @ membership.T
    with np.errstate(invalid="ignore", divide="ignore"):
        subject_means = np.where(weight > 0, total / weight, np.nan)
    ci_lo, ci_hi = (float(x) for x in np.nanpercentile(np.nanmean(subject_means, axis=1), [2.5, 97.5]))

    tasks = sorted({t for _, t in subj_task_d})
    task_delta = {t: [v for (s, tt), v in sorted(subj_task_d.items()) if tt == t] for t in tasks}
    task_p = {t: sign_flip_p(v) for t, v in task_delta.items()}
    task_holm = holm(task_p)
    per_task = {t: {"delta": float(np.mean(task_delta[t])), "p": task_p[t], "p_holm": task_holm[t]} for t in tasks}
    loso = {str(s): float(np.delete(deltas, i).mean()) for i, s in enumerate(subjects)} if len(subjects) > 1 else {}
    decision = "Descriptive" if descriptive or sessions is not None else decide(tier, regime, delta, up, len(subjects))
    return Comparison(candidate=candidate, baseline=baseline, tier=tier, regime=regime,
                      score=float(np.mean(list(subj_c.values()))),
                      baseline_score=float(np.mean(list(subj_b.values()))), delta=delta,
                      n_subjects=len(subjects), subjects_up=up, p_signflip=sign_flip_p(deltas),
                      ci_lo=ci_lo, ci_hi=ci_hi, decision=decision,
                      per_subject={str(s): subj_d[s] for s in subjects}, per_session=sess_d,
                      per_task=per_task, leave_one_subject_out=loso, n_cells=len(cand),
                      n_nan_cells=n_nan, sessions=tuple(session_names))
