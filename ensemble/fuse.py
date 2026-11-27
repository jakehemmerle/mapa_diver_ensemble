"""Score fusion over prediction files.

Rung (a): uniform mean of rank-normalized member scores.
Rung (b): non-negative logistic stacker with a fixed penalty, fit per cell on the
members' out-of-fold predictions for the training rows and their training labels.

Rank normalization is label-free: a member's select and eval scores are mapped through
the empirical distribution of that member's own select scores for the cell (same fitted
model, same target session, so the spread matches the eval scores). Its out-of-fold
training scores, which the stacker is fit on, are mapped through their own distribution;
they are 2-3x wider than held-out scores, so they cannot serve as the reference for the
held-out parts (the "train_oof_cdf" reference used before November 25, 19:40, did).
No select or eval label is read here.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import ledger, predictions

RUNGS = ("a", "b")
STACKER_PENALTY = 0.01      # fixed in advance; pulls the weights toward uniform
NORMALIZATION = "select_cdf"      # select/eval through the select scores; train through the train scores
_TAIL = 0.05


def fused_name(members, rung: str, prefix: str = "ens") -> str:
    return f"{prefix}_{rung}__" + "+".join(sorted(members))


def subsets(pool, sizes=(2, 3, 4)):
    pool = sorted(pool)
    return [combo for size in sizes for combo in itertools.combinations(pool, size)]


def normalize(scores, reference) -> np.ndarray:
    """Map scores through the empirical CDF of ``reference`` (label-free scores of the same
    member), centred and scaled to unit variance under that reference.

    The map is strictly increasing: linear between reference values and extended with
    the slope of the outer 5% beyond them, so a member's ranking is never changed.
    """
    reference = np.asarray(reference, dtype=np.float64)
    values, counts = np.unique(reference, return_counts=True)
    if len(values) < 2:
        raise ValueError("the reference needs at least two distinct scores")
    upper = np.cumsum(counts)
    levels = (upper - (counts - 1) / 2.0 - 0.5) / reference.size      # mid-rank quantile levels
    step = max(1, int(round(_TAIL * (len(values) - 1))))
    low = (levels[step] - levels[0]) / (values[step] - values[0])
    high = (levels[-1] - levels[-1 - step]) / (values[-1] - values[-1 - step])
    s = np.asarray(scores, dtype=np.float64)
    out = np.interp(s, values, levels)
    out = np.where(s < values[0], levels[0] + (s - values[0]) * low, out)
    out = np.where(s > values[-1], levels[-1] + (s - values[-1]) * high, out)
    return (out - 0.5) * np.sqrt(12.0)


def fit_stacker(x: np.ndarray, y: np.ndarray, penalty: float = STACKER_PENALTY) -> np.ndarray:
    """Weights on the simplex for ``sigmoid(a * x @ w + b)`` with ``a >= 0``.

    Minimizes mean log loss plus ``penalty * M * ||w - 1/M||^2``. With no signal the
    scale ``a`` goes to zero and the penalty returns the uniform mean.
    """
    x = np.asarray(x, dtype=np.float64)
    y = (np.asarray(y) > 0).astype(np.float64)
    n, m = x.shape
    if y.shape != (n,):
        raise ValueError("stacker inputs and labels differ in length")
    if y.min() == y.max():
        return np.full(m, 1.0 / m)
    prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)

    def loss(theta):
        w, a, b = theta[:m], theta[m], theta[m + 1]
        z = a * (x @ w) + b
        p = 0.5 * (1.0 + np.tanh(0.5 * z))
        value = np.mean(np.logaddexp(0.0, z) - y * z) + penalty * m * np.sum((w - 1.0 / m) ** 2)
        residual = (p - y) / n
        grad_w = a * (x.T @ residual) + 2.0 * penalty * m * (w - 1.0 / m)
        return value, np.r_[grad_w, residual @ (x @ w), residual.sum()]

    start = np.r_[np.full(m, 1.0 / m), 1.0, np.log(prior / (1 - prior))]
    result = minimize(loss, start, jac=True, method="SLSQP",
                      bounds=[(0.0, 1.0)] * m + [(0.0, 50.0), (-20.0, 20.0)],
                      constraints=[{"type": "eq", "fun": lambda t: t[:m].sum() - 1.0,
                                    "jac": lambda t: np.r_[np.ones(m), 0.0, 0.0]}],
                      options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(result.x[:m], 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise RuntimeError(f"stacker did not converge: {result.message}")
    return w / w.sum()


def _cell_seed(cell, seed: int) -> int:
    digest = hashlib.sha256(f"{cell.tier}/{cell.regime}/{cell.key()}/{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def fuse_cell(cell, members, rung: str, root: Path, *, splits=None, name: str | None = None,
              train_reference: dict | None = None, shuffle_seed: int | None = None,
              penalty: float = STACKER_PENALTY) -> Path:
    """Fuse ``members`` for one cell and write the result as a prediction file.

    ``train_reference`` maps a member to the member whose out-of-fold training scores
    serve as its reference (used for seed replicates, which carry no training part).
    ``shuffle_seed`` permutes the training labels before the stacker fit (control).
    """
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}")
    members = list(members)
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("fusion needs at least two distinct members")
    train_reference = train_reference or {}
    preds = {m: predictions.read(cell, m, root) for m in members}
    refs = {}
    for m in members:
        source = train_reference.get(m, m)
        ref = preds[m] if source == m else predictions.read(cell, source, root)
        if "train" not in ref.scores and rung == "b":
            raise KeyError(f"{source} has no out-of-fold training scores for {cell.key()}")
        refs[m] = ref
    first = members[0]
    normalized, rows = {}, {}
    for part in ("select", "eval"):
        rows[part] = preds[first].rows[part]
        normalized[part] = np.column_stack([
            normalize(predictions.align(preds[m].scores[part], preds[m].rows[part], rows[part],
                                        f"{m} vs {first}, {cell.key()} {part}"), preds[m].scores["select"])
            for m in members])
    shared_train = all("train" in refs[m].scores for m in members)    # rung (a) needs no training part
    try:
        if shared_train:
            rows["train"] = refs[first].rows["train"]
            normalized["train"] = np.column_stack([
                normalize(predictions.align(refs[m].scores["train"], refs[m].rows["train"], rows["train"],
                                            f"{m} vs {first}, {cell.key()} train"), refs[m].scores["train"])
                for m in members])
    except predictions.RowSetMismatch:
        if rung == "b":
            raise
        shared_train = False
        rows.pop("train", None)
    meta = {"members": members, "rung": rung, "normalization": NORMALIZATION,
            "commit": ledger.commit(), "train_reference": train_reference}
    if rung == "a":
        weights = np.full(len(members), 1.0 / len(members))
    else:
        if splits is None:
            from common import splits as splits_module
            splits = splits_module
        split = splits.get_cell(cell)
        y = predictions.align(np.asarray(split.labels("train")), np.asarray(split.rows["train"]), rows["train"],
                              f"split vs {first}, {cell.key()} train")
        if shuffle_seed is not None:
            y = np.random.default_rng(_cell_seed(cell, shuffle_seed)).permutation(y)
            meta["shuffled_training_labels"] = int(shuffle_seed)
        weights = fit_stacker(normalized["train"], y, penalty)
        meta["penalty"] = penalty
    meta["weights"] = dict(zip(members, weights.tolist()))
    meta["config_hash"] = ledger.config_hash({k: meta[k] for k in ("members", "rung", "normalization")}
                                             | ({"penalty": penalty} if rung == "b" else {}))
    scores = {part: normalized[part] @ weights for part in normalized}
    if not shared_train:
        scores.pop("train", None)
    return predictions.write(cell, name or fused_name(members, rung), rows, scores, meta, root)


def fuse(members, rung: str, tier: str, regime: str, root: Path, *, splits=None,
         name: str | None = None, sessions=None, **options) -> str:
    """Fuse ``members`` on every cell of a tier and regime; return the fused member name."""
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    name = name or fused_name(members, rung)
    for cell in splits.list_cells(tier, regime):
        if sessions is None or cell.target_session in sessions:
            fuse_cell(cell, members, rung, root, splits=splits, name=name, **options)
    return name
